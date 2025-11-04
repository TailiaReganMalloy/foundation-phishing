from pathlib import Path

import pandas as pd
from scipy.io import savemat

DATA_DIR = Path(__file__).resolve().parent
SUPPORTED_LOADERS = {
	".pkl": pd.read_pickle,
	".csv": pd.read_csv,
	".json": pd.read_json,
	".xlsx": pd.read_excel,
}


def ensure_dataframe(obj: object) -> pd.DataFrame:
	"""Convert loaded objects to DataFrame so all exports are supported."""
	if isinstance(obj, pd.DataFrame):
		return obj
	if isinstance(obj, pd.Series):
		return obj.to_frame()
	return pd.DataFrame(obj)


def load_dataset(folder: Path) -> tuple[pd.DataFrame, Path]:
	"""Load the first supported file in a folder and return the DataFrame and its stem."""
	for candidate in sorted(folder.iterdir()):
		loader = SUPPORTED_LOADERS.get(candidate.suffix.lower())
		if loader is None:
			continue
		dataset = ensure_dataframe(loader(candidate))
		stem = candidate.with_suffix("")
		return dataset, stem
	raise FileNotFoundError(f"No supported data files found in {folder}")


def export_formats(df: pd.DataFrame, base_path: Path) -> None:
	"""Write the dataset to CSV, PKL, JSON, XLSX, and MAT formats."""
	df.to_csv(base_path.with_suffix(".csv"), index=False)
	df.to_pickle(base_path.with_suffix(".pkl"))
	df.to_json(base_path.with_suffix(".json"), orient="records")
	df.to_excel(base_path.with_suffix(".xlsx"), index=False)
	savemat(
		base_path.with_suffix(".mat"),
		{
			"data": df.to_numpy(dtype=object),
			"columns": df.columns.to_numpy(dtype=object),
		},
	)


def process_folder(folder_name: str) -> None:
	folder = DATA_DIR / folder_name
	dataset, stem = load_dataset(folder)
	export_formats(dataset, folder / stem.name)


def main() -> None:
	for name in ("Emails", "Embeddings", "Questions", "Responses"):
		process_folder(name)


if __name__ == "__main__":
	main()