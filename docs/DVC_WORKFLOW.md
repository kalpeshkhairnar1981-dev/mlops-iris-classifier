# DVC Workflow for the Iris Classifier

## Remote configuration

This project uses a local DVC remote to simulate a shared storage backend:

- Remote name: `local`
- Remote path: `C:\Users\Admin\dvc-remote-storage`

The configuration is stored in `.dvc/config` and is committed to Git so that every collaborator points to the same storage backend.

## DVC data lifecycle

For every dataset change, the workflow is:

1. Generate or update the dataset file.
2. Run `dvc add data/raw/iris_v1.csv`.
3. Stage the DVC metadata with `git add data/raw/.gitignore data/raw/iris_v1.csv.dvc`.
4. Commit the metadata with Git: `git commit -m "data: ..."`.
5. Upload the object to the remote with `dvc push`.

This keeps Git tracking only the small DVC pointer files while the actual CSV content remains in the DVC cache and remote.

## Dataset comparison and restore

- `git log --oneline -- data/raw/iris_v1.csv.dvc` shows the dataset version history.
- `dvc diff <commit>` compares the current DVC-tracked dataset against a previous hash.
- `git checkout <commit> -- data/raw/iris_v1.csv.dvc` restores the previous pointer.
- `dvc checkout data/raw/iris_v1.csv.dvc` restores the actual data file corresponding to that pointer.

This workflow allows the project to switch between dataset versions reliably without storing full copies in Git.

## Example history used in this project

The raw Iris dataset was first stored at 150 rows and then augmented to 170 rows. DVC recorded a new object hash for the second version while preserving the first version in the remote and cache.

The `dvc diff` and `dvc checkout` commands were used to confirm that both versions could be compared and restored on demand.
