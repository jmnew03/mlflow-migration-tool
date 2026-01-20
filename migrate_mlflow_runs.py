#!/usr/bin/env python3
"""
MLflow Run Migration Tool

Migrates runs from a source experiment to a target experiment,
preserving all parameters, metrics, tags, and optionally artifacts.

High-performance migration using batch API calls and parallel processing.
"""

import argparse
import logging
import sys
import time
from typing import List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import mlflow
    from mlflow.entities import RunStatus, Metric, Param
    from mlflow.tracking import MlflowClient
except ImportError:
    print("Error: mlflow is not installed. Install it with: pip install mlflow")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLflowRunMigrator:
    """Handles migration of MLflow runs between experiments."""

    def __init__(self, tracking_uri: Optional[str] = None, batch_size: int = 1000):
        """
        Initialize the migrator.

        Args:
            tracking_uri: MLflow tracking server URI (None uses default)
            batch_size: Number of metrics to log in each batch (default: 1000)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()
        self.batch_size = batch_size

    def validate_experiment(self, experiment_id: str) -> bool:
        """Validate that an experiment exists."""
        try:
            exp = self.client.get_experiment(experiment_id)
            if exp is None:
                logger.error(f"Experiment {experiment_id} not found")
                return False
            logger.info(f"✓ Experiment '{exp.name}' (ID: {experiment_id}) found")
            return True
        except Exception as e:
            logger.error(f"Error validating experiment {experiment_id}: {e}")
            return False

    def get_runs(self, experiment_id: str, run_ids: Optional[List[str]] = None,
                 filter_tags: Optional[dict] = None) -> List:
        """
        Get all runs from an experiment with optional filtering.

        Args:
            experiment_id: Source experiment ID
            run_ids: Optional list of specific run IDs to migrate
            filter_tags: Optional dict of tags to filter runs

        Returns:
            List of run objects
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["start_time DESC"]
            )

            # Filter by run IDs if specified
            if run_ids:
                runs = [r for r in runs if r.info.run_id in run_ids]

            # Filter by tags if specified
            if filter_tags:
                runs = [
                    r for r in runs
                    if all(r.data.tags.get(k) == v for k, v in filter_tags.items())
                ]

            logger.info(f"Found {len(runs)} run(s) to migrate")
            return runs
        except Exception as e:
            logger.error(f"Error retrieving runs: {e}")
            return []

    def migrate_run(self, source_run, target_experiment_id: str,
                   migrate_artifacts: bool = False) -> Optional[str]:
        """
        Migrate a single run to the target experiment.

        Args:
            source_run: Source run object
            target_experiment_id: Target experiment ID
            migrate_artifacts: Whether to migrate artifacts

        Returns:
            New run ID if successful, None otherwise
        """
        run_id = source_run.info.run_id
        run_name = source_run.data.tags.get('mlflow.runName', f'run_{run_id[:8]}')

        logger.info(f"\n{'='*60}")
        logger.info(f"Migrating run: {run_name} ({run_id})")

        try:
            # Create new run in target experiment
            new_run = self.client.create_run(
                experiment_id=target_experiment_id,
                start_time=source_run.info.start_time,
                tags={
                    **source_run.data.tags,
                    'migrated_from_run_id': run_id,
                    'migrated_from_experiment_id': source_run.info.experiment_id,
                    'migration_timestamp': str(datetime.now())
                }
            )
            new_run_id = new_run.info.run_id
            logger.info(f"  ✓ Created new run: {new_run_id}")

            # Migrate parameters
            params_count = self._migrate_parameters(source_run, new_run_id)
            logger.info(f"  ✓ Migrated {params_count} parameter(s)")

            # Migrate metrics
            metrics_count = self._migrate_metrics(source_run, new_run_id)
            logger.info(f"  ✓ Migrated {metrics_count} metric(s)")

            # Migrate artifacts if requested
            if migrate_artifacts:
                artifacts_count = self._migrate_artifacts(source_run, new_run_id)
                logger.info(f"  ✓ Migrated {artifacts_count} artifact(s)")

            # Update run status and end time if completed
            if source_run.info.status != RunStatus.to_string(RunStatus.RUNNING):
                self.client.set_terminated(
                    new_run_id,
                    status=source_run.info.status,
                    end_time=source_run.info.end_time
                )
                logger.info(f"  ✓ Set run status: {source_run.info.status}")

            logger.info(f"  ✅ Migration complete for run {run_name}")
            return new_run_id

        except Exception as e:
            logger.error(f"  ❌ Error migrating run {run_id}: {e}")
            return None

    def _migrate_parameters(self, source_run, new_run_id: str) -> int:
        """Migrate all parameters from source run to new run using batch API."""
        params = source_run.data.params
        if not params:
            return 0

        # Use log_batch for efficient parameter logging
        param_list = [Param(key, value) for key, value in params.items()]
        self.client.log_batch(new_run_id, params=param_list)
        return len(params)

    def _migrate_metrics(self, source_run, new_run_id: str) -> int:
        """Migrate all metrics from source run to new run using batch API."""
        run_id = source_run.info.run_id
        metric_count = 0

        # Collect all metrics first
        all_metrics = []
        for metric_key in source_run.data.metrics.keys():
            # Get metric history to preserve timestamps and steps
            metric_history = self.client.get_metric_history(run_id, metric_key)
            all_metrics.extend(metric_history)

        if not all_metrics:
            return 0

        # Log metrics in batches for efficiency
        for i in range(0, len(all_metrics), self.batch_size):
            batch = all_metrics[i:i + self.batch_size]
            self.client.log_batch(new_run_id, metrics=batch)
            metric_count += len(batch)

        return metric_count

    def _migrate_artifacts(self, source_run, new_run_id: str) -> int:
        """Migrate artifacts from source run to new run."""
        try:
            artifacts = self.client.list_artifacts(source_run.info.run_id)

            for artifact in artifacts:
                # Download artifact
                local_path = self.client.download_artifacts(
                    source_run.info.run_id,
                    artifact.path
                )

                # Upload to new run
                self.client.log_artifact(new_run_id, local_path)

            return len(artifacts)
        except Exception as e:
            logger.warning(f"  ⚠ Could not migrate artifacts: {e}")
            return 0

    def migrate_experiment(self, source_experiment_id: str,
                          target_experiment_id: str,
                          run_ids: Optional[List[str]] = None,
                          filter_tags: Optional[dict] = None,
                          migrate_artifacts: bool = False,
                          parallel_workers: int = 1) -> dict:
        """
        Migrate all runs from source to target experiment.

        Args:
            source_experiment_id: Source experiment ID
            target_experiment_id: Target experiment ID
            run_ids: Optional list of specific run IDs to migrate
            filter_tags: Optional dict of tags to filter runs
            migrate_artifacts: Whether to migrate artifacts
            parallel_workers: Number of parallel workers (default: 1 for sequential)

        Returns:
            Dict with migration statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info("MLflow Run Migration")
        logger.info(f"{'='*60}")
        logger.info(f"Source Experiment: {source_experiment_id}")
        logger.info(f"Target Experiment: {target_experiment_id}")

        # Validate experiments
        if not self.validate_experiment(source_experiment_id):
            return {'success': False, 'error': 'Invalid source experiment'}

        if not self.validate_experiment(target_experiment_id):
            return {'success': False, 'error': 'Invalid target experiment'}

        # Get runs to migrate
        runs = self.get_runs(source_experiment_id, run_ids, filter_tags)

        if not runs:
            logger.warning("No runs found to migrate")
            return {'success': True, 'migrated': 0, 'failed': 0}

        # Start timing
        start_time = time.time()

        # Migrate each run (sequential or parallel)
        migrated_count = 0
        failed_count = 0

        if parallel_workers > 1:
            logger.info(f"Using {parallel_workers} parallel workers")
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                # Submit all migration tasks
                future_to_run = {
                    executor.submit(
                        self.migrate_run, run, target_experiment_id, migrate_artifacts
                    ): run for run in runs
                }

                # Collect results as they complete
                for future in as_completed(future_to_run):
                    result = future.result()
                    if result:
                        migrated_count += 1
                    else:
                        failed_count += 1
        else:
            # Sequential processing
            for run in runs:
                result = self.migrate_run(run, target_experiment_id, migrate_artifacts)
                if result:
                    migrated_count += 1
                else:
                    failed_count += 1

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Migration Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total runs processed: {len(runs)}")
        logger.info(f"Successfully migrated: {migrated_count}")
        if failed_count > 0:
            logger.info(f"Failed: {failed_count}")
        logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        if migrated_count > 0:
            logger.info(f"Average time per run: {elapsed_time/migrated_count:.2f} seconds")
        logger.info(f"{'='*60}\n")

        return {
            'success': True,
            'migrated': migrated_count,
            'failed': failed_count,
            'total': len(runs),
            'elapsed_time': elapsed_time
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Migrate MLflow runs from one experiment to another (high-performance)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic migration
  python migrate_mlflow_runs.py --source 1 --target 2

  # Fast migration with parallel workers (recommended)
  python migrate_mlflow_runs.py --source 1 --target 2 --parallel 4

  # Migrate specific runs
  python migrate_mlflow_runs.py --source 1 --target 2 --run-ids abc123,def456

  # Migrate with artifacts
  python migrate_mlflow_runs.py --source 1 --target 2 --artifacts

  # Maximum speed (large batches + parallel)
  python migrate_mlflow_runs.py --source 1 --target 2 --batch-size 2000 --parallel 8

  # Custom tracking server
  python migrate_mlflow_runs.py --source 1 --target 2 --tracking-uri http://localhost:5000
        """
    )

    parser.add_argument(
        '--source', '-s',
        required=True,
        help='Source experiment ID'
    )

    parser.add_argument(
        '--target', '-t',
        required=True,
        help='Target experiment ID'
    )

    parser.add_argument(
        '--run-ids',
        help='Comma-separated list of specific run IDs to migrate'
    )

    parser.add_argument(
        '--filter-tags',
        help='Filter runs by tags (format: key1=val1,key2=val2)'
    )

    parser.add_argument(
        '--artifacts',
        action='store_true',
        help='Migrate artifacts (may be slow for large artifacts)'
    )

    parser.add_argument(
        '--tracking-uri',
        help='MLflow tracking server URI (default: uses MLFLOW_TRACKING_URI env var)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of metrics to log per batch (default: 1000). Larger = faster but more memory'
    )

    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=1,
        help='Number of parallel workers for run migration (default: 1). Use 4-8 for faster migration'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Parse run IDs if provided
    run_ids = None
    if args.run_ids:
        run_ids = [rid.strip() for rid in args.run_ids.split(',')]

    # Parse filter tags if provided
    filter_tags = None
    if args.filter_tags:
        filter_tags = {}
        for pair in args.filter_tags.split(','):
            key, value = pair.split('=')
            filter_tags[key.strip()] = value.strip()

    # Initialize migrator
    migrator = MLflowRunMigrator(
        tracking_uri=args.tracking_uri,
        batch_size=args.batch_size
    )

    # Perform migration
    result = migrator.migrate_experiment(
        source_experiment_id=args.source,
        target_experiment_id=args.target,
        run_ids=run_ids,
        filter_tags=filter_tags,
        migrate_artifacts=args.artifacts,
        parallel_workers=args.parallel
    )

    # Exit with appropriate code
    if result['success']:
        sys.exit(0)
    else:
        logger.error(f"Migration failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
