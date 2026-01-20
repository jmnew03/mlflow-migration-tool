# MLflow Run Migration Tool ⚡

A high-performance Python script to migrate MLflow runs between experiments in a server. 

## ✨ Features

- ⚡ **Parallel processing** for multi-run migrations
- 📦 **Batch operations** for efficient data transfer
- ✅ **Complete data preservation**: parameters, metrics (with history), tags, artifacts
- 🔍 **Selective migration**: filter by run IDs or tags
- 🛡️ **Non-destructive**: original runs remain untouched

## 🚀 Quick Start

### Installation

```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/mlflow-migration-tool.git
cd mlflow-migration-tool

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Migrate all runs from experiment 1 to experiment 2
python migrate_mlflow_runs.py --source 1 --target 2
```

### Fast Migration (Recommended)

```bash
# Use 4 parallel workers for maximum speed
python migrate_mlflow_runs.py --source 1 --target 2 --parallel 4
```

## 📖 Usage Examples

### Migrate Specific Runs

```bash
python migrate_mlflow_runs.py --source 1 --target 2 --run-ids abc123,def456
```

### Filter by Tags

```bash
python migrate_mlflow_runs.py --source 1 --target 2 --filter-tags env=prod,version=v2
```

### Include Artifacts

```bash
python migrate_mlflow_runs.py --source 1 --target 2 --artifacts
```

### Custom MLflow Server

```bash
# Via command line
python migrate_mlflow_runs.py --source 1 --target 2 --tracking-uri http://mlflow-server:5000

# Or set environment variable
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
python migrate_mlflow_runs.py --source 1 --target 2
```

### Maximum Performance

```bash
# Large batch size + 8 parallel workers
python migrate_mlflow_runs.py --source 1 --target 2 --batch-size 2000 --parallel 8
```

## 🎯 Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--source`, `-s` | ✅ | Source experiment ID |
| `--target`, `-t` | ✅ | Target experiment ID |
| `--run-ids` | ❌ | Comma-separated list of specific run IDs |
| `--filter-tags` | ❌ | Filter by tags (format: `key1=val1,key2=val2`) |
| `--artifacts` | ❌ | Include artifact migration (slower) |
| `--tracking-uri` | ❌ | MLflow tracking server URI |
| `--batch-size` | ❌ | Metrics per batch (default: 1000) |
| `--parallel`, `-p` | ❌ | Number of parallel workers (default: 1) |
| `--verbose`, `-v` | ❌ | Enable verbose logging |

## 💡 Finding Your Experiment IDs

### Option 1: MLflow UI
Open your MLflow UI and check the URL:
```
http://your-server:5000/#/experiments/123
                                      ^^^
                                  Experiment ID
```

### Option 2: Command Line
```bash
mlflow experiments list
```

### Option 3: Python
```python
import mlflow
client = mlflow.tracking.MlflowClient()
for exp in client.search_experiments():
    print(f"ID: {exp.experiment_id}, Name: {exp.name}")
```

## 🔐 Authentication

The script uses the same authentication as your MLflow CLI/UI:

### No Authentication (default MLflow)
```bash
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
python migrate_mlflow_runs.py --source 1 --target 2
```

### Basic Authentication
```bash
export MLFLOW_TRACKING_URI="http://user:pass@mlflow-server:5000"
python migrate_mlflow_runs.py --source 1 --target 2
```

### Token-Based (Databricks, etc.)
```bash
export MLFLOW_TRACKING_URI="databricks"
export DATABRICKS_TOKEN="dapi..."
python migrate_mlflow_runs.py --source 1 --target 2
```

## 📊 Performance

### Performance Tips

- **Use `--parallel 4` to `--parallel 8`** for experiments with many runs
- **Increase `--batch-size` to 2000-5000** for runs with many metrics
- **Avoid `--artifacts`** unless necessary (artifacts are slow to transfer)

## 🔧 How It Works

1. **Reads** all data from source experiment runs
2. **Creates** new runs in target experiment
3. **Copies** parameters, metrics (with timestamps), and tags using batch API
4. **Preserves** all metric history with correct timestamps and step numbers
5. **Adds** migration metadata tags for traceability

### What Gets Migrated?

- ✅ Run metadata (name, start/end time, status)
- ✅ All parameters
- ✅ All metrics with complete history (timestamps + steps)
- ✅ All tags (+ migration provenance tags)
- ✅ Artifacts (optional, via `--artifacts` flag)

### Migration Tags

Each migrated run includes these tags:
- `migrated_from_run_id`: Original run ID
- `migrated_from_experiment_id`: Original experiment ID
- `migration_timestamp`: When migration occurred

## 🛡️ Safety

- **Non-destructive**: Original runs are never modified or deleted
- **New run IDs**: Migrated runs get new IDs in the target experiment
- **Traceable**: Migration tags allow tracking back to originals

## 📝 Requirements

- Python 3.7+
- MLflow 2.0+
