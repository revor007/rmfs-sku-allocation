from __future__ import annotations

import re
from difflib import SequenceMatcher
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent

RAW_PRODUCT_DATA_CANDIDATES = [
    WORKSPACE_DIR / "Webscrapping" / "webscraping_result.csv",
]
MASTER_PRODUCT_DATA_CANDIDATES = [
    BASE_DIR / "儲格設計_原檔(商品資訊).csv",
    WORKSPACE_DIR / "fcgma" / "Preprocessing" / "儲格設計_原檔(商品資訊).csv",
]
RAW_ORDER_DATA_CANDIDATES = [
    BASE_DIR / "訂單資料(order data).csv",
    WORKSPACE_DIR / "fcgma" / "Preprocessing" / "訂單資料(order data).csv",
]

PRODUCT_OUTPUT_PATH = BASE_DIR / "preprocessed_final.csv"
ORDER_OUTPUT_PATH = BASE_DIR / "訂單資料_final.csv"
ANALYSIS_SUMMARY_PATH = BASE_DIR / "data_cleaning_analysis_summary.csv"
VERIFICATION_SUMMARY_PATH = BASE_DIR / "data_cleaning_verification_summary.csv"
FILTERED_DATA_DIR = BASE_DIR / "Filtered Data"

MASTER_ITEM_CODE_CANDIDATES = ["item_code", "Item Code"]
MASTER_QUERY_CANDIDATES = ["註記", "註記 (Notes)"]
MASTER_CARTON_LENGTH_CANDIDATES = ["(箱)長cm", "(箱)長 (Length Carton) cm"]
MASTER_CARTON_WIDTH_CANDIDATES = ["(箱)寬cm", "(箱)寬 (Width) cm"]
MASTER_CARTON_HEIGHT_CANDIDATES = ["(箱)高cm", "(箱)高 (Heigth) cm"]
MASTER_CARTON_WEIGHT_CANDIDATES = ["(箱)重量", "(箱)重量 (Weigth)"]
MASTER_UNITS_PER_CARTON_CANDIDATES = ["箱入數", "箱入數 (Number of Cartons)"]

ORDER_ID_CANDIDATES = ["订单号", "order_id"]
ORDER_ITEM_CODE_CANDIDATES = ["商品编码", "item_code"]
ORDER_ITEM_NAME_CANDIDATES = ["商品名称", "item_name"]
ORDER_QUANTITY_CANDIDATES = ["商品数量", "quantity"]
ORDER_CREATED_AT_CANDIDATES = ["创建时间", "created_at"]

FILTER_EXPORTS = [
    ("飲料零食", ["volume", "weight", "count"]),
    ("傢俱寢飾", ["count"]),
    ("大家都買這些", ["weight"]),
    ("嬰童保健", ["volume", "weight", "count"]),
    ("日用生活", ["volume", "weight", "count", "length"]),
    ("服飾鞋包", ["count"]),
    ("熱門3C", ["count"]),
    ("生活家電", ["count", "volume"]),
    ("生鮮冷凍", ["count", "volume", "weight"]),
    ("米油沖泡", ["count", "weight", "volume"]),
    ("美妝個清", ["volume", "weight", "count", "length"]),
]

MASTER_DIMENSION_RENAME_MAP = {
    "carton_length_raw": "carton_length_cm",
    "carton_width_raw": "carton_width_cm",
    "carton_height_raw": "carton_height_cm",
    "carton_weight_raw": "carton_weight",
    "units_per_carton_raw": "units_per_carton",
}

COUNT_UNIT_MAP = {
    "入組": "set",
    "組": "set",
    "入": "pc",
    "個": "pc",
    "件": "pc",
    "支": "pc",
    "張": "sheet",
    "雙": "pair",
    "捲": "roll",
    "卡": "card",
    "包": "pack",
    "袋": "bag",
    "盒": "box",
    "箱": "box",
    "瓶": "bottle",
    "罐": "can",
    "桶": "bucket",
    "碗": "bowl",
}
COUNT_UNIT_PATTERN = "|".join(
    re.escape(unit) for unit in sorted(COUNT_UNIT_MAP, key=len, reverse=True)
)

UNIT_SYNONYMS = {
    "pcs": "pc",
    "piece": "pc",
    "pieces": "pc",
    "btl": "bottle",
    "bottle": "bottle",
    "bottles": "bottle",
    "bot": "bottle",
    "pack": "pack",
    "packs": "pack",
    "pkg": "pack",
    "bag": "bag",
    "bags": "bag",
    "box": "box",
    "boxes": "box",
    "set": "set",
    "sets": "set",
    "can": "can",
    "cans": "can",
    "roll": "roll",
    "rolls": "roll",
    "pair": "pair",
    "pairs": "pair",
    "card": "card",
    "cards": "card",
    "bucket": "bucket",
    "buckets": "bucket",
    "bowl": "bowl",
    "bowls": "bowl",
    "ctn": "box",
}


def find_existing_path(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked: {checked}")


def normalize_column_name(name: object) -> str:
    return str(name).replace("\ufeff", "").strip()


def find_column(columns: pd.Index, candidates: list[str]) -> str:
    normalized = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    raise KeyError(
        f"Could not find any of the expected columns {candidates}. "
        f"Available columns: {list(columns)}"
    )


def normalize_item_code(value: object) -> str:
    if pd.isna(value):
        return ""

    text = normalize_column_name(value)
    if not text or text.lower() in {"nan", "none"}:
        return ""

    try:
        numeric_value = float(text)
    except ValueError:
        return text

    if isfinite(numeric_value) and numeric_value.is_integer():
        return str(int(numeric_value))
    return text


def normalize_lookup_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).replace("\ufeff", "").replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_similarity_text(value: object) -> str:
    text = normalize_lookup_text(value).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def similarity_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def get_writable_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    try:
        with open(path, "a", encoding="utf-8"):
            return path
    except PermissionError:
        return path.with_name(f"{path.stem}_latest{path.suffix}")


def save_dataframe(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    output_path = get_writable_output_path(path)
    df.to_csv(output_path, **kwargs)
    return output_path


def safe_console_text(value: object) -> str:
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def load_raw_product_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        engine="python",
    ).copy()


def load_master_product_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        engine="python",
    ).copy()


def load_raw_order_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        engine="python",
    ).copy()


def patternize_text(value: object) -> str:
    text = normalize_lookup_text(value)
    if not text:
        return ""

    text = re.sub(r"\d", "9", text)
    text = re.sub(r"[A-Za-z]+", "a", text)
    text = re.sub(r"[\u4e00-\u9fff]+", "中", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_profile_records(
    records: list[dict[str, object]],
    dataset_name: str,
    df: pd.DataFrame,
) -> None:
    records.append(
        {
            "section": "profile",
            "dataset": dataset_name,
            "column": "",
            "metric": "row_count",
            "value": len(df),
        }
    )
    records.append(
        {
            "section": "profile",
            "dataset": dataset_name,
            "column": "",
            "metric": "column_count",
            "value": len(df.columns),
        }
    )

    for column in df.columns:
        series = df[column]
        clean = series.dropna()
        records.extend(
            [
                {
                    "section": "profile",
                    "dataset": dataset_name,
                    "column": column,
                    "metric": "dtype",
                    "value": str(series.dtype),
                },
                {
                    "section": "profile",
                    "dataset": dataset_name,
                    "column": column,
                    "metric": "null_count",
                    "value": int(series.isna().sum()),
                },
                {
                    "section": "profile",
                    "dataset": dataset_name,
                    "column": column,
                    "metric": "unique_count",
                    "value": int(clean.nunique(dropna=True)),
                },
            ]
        )

        if not clean.empty:
            string_lengths = clean.astype(str).str.len()
            records.extend(
                [
                    {
                        "section": "profile",
                        "dataset": dataset_name,
                        "column": column,
                        "metric": "min_length",
                        "value": int(string_lengths.min()),
                    },
                    {
                        "section": "profile",
                        "dataset": dataset_name,
                        "column": column,
                        "metric": "max_length",
                        "value": int(string_lengths.max()),
                    },
                    {
                        "section": "profile",
                        "dataset": dataset_name,
                        "column": column,
                        "metric": "top_patterns",
                        "value": " | ".join(
                            f"{pattern}:{count}"
                            for pattern, count in clean.astype(str)
                            .map(patternize_text)
                            .replace("", np.nan)
                            .dropna()
                            .value_counts()
                            .head(3)
                            .items()
                        ),
                    },
                ]
            )

        numeric = pd.to_numeric(series, errors="coerce")
        numeric = numeric.dropna()
        if not numeric.empty:
            records.extend(
                [
                    {
                        "section": "profile",
                        "dataset": dataset_name,
                        "column": column,
                        "metric": "numeric_min",
                        "value": float(numeric.min()),
                    },
                    {
                        "section": "profile",
                        "dataset": dataset_name,
                        "column": column,
                        "metric": "numeric_max",
                        "value": float(numeric.max()),
                    },
                    {
                        "section": "profile",
                        "dataset": dataset_name,
                        "column": column,
                        "metric": "numeric_variance",
                        "value": float(numeric.var(ddof=0)),
                    },
                ]
            )


def analyze_raw_datasets(
    product_df: pd.DataFrame,
    master_df: pd.DataFrame,
    order_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    records: list[dict[str, object]] = []
    add_profile_records(records, "product_raw", product_df)
    add_profile_records(records, "master_raw", master_df)
    add_profile_records(records, "order_raw", order_df)

    master_item_code_col = find_column(master_df.columns, MASTER_ITEM_CODE_CANDIDATES)
    master_query_col = find_column(master_df.columns, MASTER_QUERY_CANDIDATES)
    order_id_col = find_column(order_df.columns, ORDER_ID_CANDIDATES)
    order_item_code_col = find_column(order_df.columns, ORDER_ITEM_CODE_CANDIDATES)
    order_item_name_col = find_column(order_df.columns, ORDER_ITEM_NAME_CANDIDATES)
    order_quantity_col = find_column(order_df.columns, ORDER_QUANTITY_CANDIDATES)
    order_created_at_col = find_column(order_df.columns, ORDER_CREATED_AT_CANDIDATES)

    raw_lookup = (
        master_df.drop_duplicates(master_query_col)
        .set_index(master_query_col)[master_item_code_col]
    )
    trimmed_master = master_df.assign(
        _query_key=master_df[master_query_col].map(normalize_lookup_text),
        _item_code_key=master_df[master_item_code_col].map(normalize_item_code),
    )
    trimmed_lookup = (
        trimmed_master.drop_duplicates("_query_key")
        .set_index("_query_key")["_item_code_key"]
    )

    raw_product_match = product_df["query"].map(raw_lookup)
    trimmed_product_match = product_df["query"].map(normalize_lookup_text).map(trimmed_lookup)
    master_note_item_code_counts = (
        trimmed_master.groupby("_query_key")["_item_code_key"].nunique()
    )
    ambiguous_note_keys = set(master_note_item_code_counts[master_note_item_code_counts.gt(1)].index)

    trimmed_product_queries = product_df["query"].map(normalize_lookup_text)
    trimmed_product_codes = set(
        code for code in trimmed_product_match.map(normalize_item_code) if code
    )
    raw_order_codes = set(order_df[order_item_code_col].map(normalize_item_code))
    parsed_timestamps = pd.to_datetime(
        order_df[order_created_at_col],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    order_quantities = pd.to_numeric(order_df[order_quantity_col], errors="coerce")

    issue_counts = {
        "product_duplicate_titles": int(product_df.duplicated(subset=["title"]).sum()),
        "product_duplicate_queries": int(product_df.duplicated(subset=["query"]).sum()),
        "product_unmatched_queries_before_trim": int(raw_product_match.isna().sum()),
        "product_unmatched_queries_after_trim": int(trimmed_product_match.map(normalize_item_code).eq("").sum()),
        "master_duplicate_query_keys": int(trimmed_master.duplicated(subset=["_query_key"]).sum()),
        "master_duplicate_item_codes": int(trimmed_master.duplicated(subset=["_item_code_key"]).sum()),
        "master_ambiguous_query_keys": int(master_note_item_code_counts.gt(1).sum()),
        "product_rows_using_ambiguous_query_keys": int(trimmed_product_queries.isin(ambiguous_note_keys).sum()),
        "order_exact_duplicate_rows": int(order_df.duplicated().sum()),
        "order_duplicate_order_code_time_rows": int(
            order_df.duplicated(
                subset=[order_id_col, order_item_code_col, order_created_at_col]
            ).sum()
        ),
        "order_invalid_timestamps": int(parsed_timestamps.isna().sum()),
        "order_nonpositive_quantities": int(order_quantities.le(0).fillna(True).sum()),
        "order_item_codes_with_multiple_names": int(
            order_df.groupby(order_item_code_col)[order_item_name_col].nunique().gt(1).sum()
        ),
        "order_item_names_with_multiple_codes": int(
            order_df.groupby(order_item_name_col)[order_item_code_col].nunique().gt(1).sum()
        ),
        "product_unique_item_codes_pre_alignment": len(trimmed_product_codes),
        "order_unique_item_codes_pre_alignment": len(raw_order_codes),
        "shared_item_codes_pre_alignment": len(trimmed_product_codes & raw_order_codes),
        "product_only_item_codes_pre_alignment": len(trimmed_product_codes - raw_order_codes),
        "order_only_item_codes_pre_alignment": len(raw_order_codes - trimmed_product_codes),
    }

    for metric, value in issue_counts.items():
        records.append(
            {
                "section": "issue",
                "dataset": "raw_analysis",
                "column": "",
                "metric": metric,
                "value": value,
            }
        )

    return pd.DataFrame(records), issue_counts


def build_rule_summary(issue_counts: dict[str, int]) -> pd.DataFrame:
    rules = [
        (
            "trim_lookup_keys",
            issue_counts["product_unmatched_queries_before_trim"]
            > issue_counts["product_unmatched_queries_after_trim"],
            "Trim whitespace in product queries and master notes before lookup.",
        ),
        (
            "resolve_ambiguous_master_matches",
            issue_counts["product_rows_using_ambiguous_query_keys"] > 0,
            "For product queries with multiple master item codes, prefer order-supported candidates and the best name similarity.",
        ),
        (
            "split_embedded_specification_values",
            True,
            "Split specification and capacity text into structured packaging attributes.",
        ),
        (
            "standardize_measurement_units",
            True,
            "Standardize unit case and normalize convertible units such as kg to g and l to ml.",
        ),
        (
            "fill_missing_defaults",
            True,
            "Fill defensible defaults for pieces per package, original price, and shelf life.",
        ),
        (
            "remove_exact_duplicate_order_rows",
            issue_counts["order_exact_duplicate_rows"] > 0,
            "Remove exact duplicate order rows while preserving repeated order-SKU combinations with different quantities.",
        ),
        (
            "filter_unmatched_item_codes",
            issue_counts["product_only_item_codes_pre_alignment"] > 0
            or issue_counts["order_only_item_codes_pre_alignment"] > 0,
            "Keep only the shared item_code set across product and order datasets.",
        ),
        (
            "standardize_order_item_names",
            issue_counts["order_item_codes_with_multiple_names"] > 0,
            "Use the matched product query as the canonical order item name after alignment.",
        ),
    ]

    return pd.DataFrame(
        [
            {
                "section": "rule",
                "dataset": "cleaning_rules",
                "column": "",
                "metric": rule_name,
                "value": description if enabled else f"Skipped: {description}",
            }
            for rule_name, enabled, description in rules
        ]
    )


def prepare_master_product_data(
    master_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    master_item_code_col = find_column(master_df.columns, MASTER_ITEM_CODE_CANDIDATES)
    master_query_col = find_column(master_df.columns, MASTER_QUERY_CANDIDATES)
    carton_length_col = find_column(master_df.columns, MASTER_CARTON_LENGTH_CANDIDATES)
    carton_width_col = find_column(master_df.columns, MASTER_CARTON_WIDTH_CANDIDATES)
    carton_height_col = find_column(master_df.columns, MASTER_CARTON_HEIGHT_CANDIDATES)
    carton_weight_col = find_column(master_df.columns, MASTER_CARTON_WEIGHT_CANDIDATES)
    units_per_carton_col = find_column(
        master_df.columns,
        MASTER_UNITS_PER_CARTON_CANDIDATES,
    )

    prepared = master_df[
        [
            master_item_code_col,
            master_query_col,
            carton_length_col,
            carton_width_col,
            carton_height_col,
            carton_weight_col,
            units_per_carton_col,
        ]
    ].copy()
    prepared.columns = [
        "item_code",
        "master_query",
        "carton_length_raw",
        "carton_width_raw",
        "carton_height_raw",
        "carton_weight_raw",
        "units_per_carton_raw",
    ]

    prepared["item_code"] = prepared["item_code"].map(normalize_item_code)
    prepared["master_query"] = prepared["master_query"].map(normalize_lookup_text)
    for raw_col in MASTER_DIMENSION_RENAME_MAP:
        prepared[raw_col] = pd.to_numeric(prepared[raw_col], errors="coerce")

    prepared["_dimension_completeness"] = prepared[list(MASTER_DIMENSION_RENAME_MAP)].notna().sum(axis=1)
    prepared = prepared.sort_values(
        ["_dimension_completeness", "item_code"],
        ascending=[False, True],
    ).reset_index(drop=True)

    master_by_item_code = prepared.drop_duplicates(subset=["item_code"], keep="first").copy()
    master_by_item_code = master_by_item_code.rename(columns=MASTER_DIMENSION_RENAME_MAP)

    master_note_groups = {
        query_key: group.copy()
        for query_key, group in prepared.groupby("master_query", sort=False)
    }
    return master_by_item_code, master_note_groups


def clean_order_dataset(order_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[str, list[str]]]:
    order_id_col = find_column(order_df.columns, ORDER_ID_CANDIDATES)
    order_item_code_col = find_column(order_df.columns, ORDER_ITEM_CODE_CANDIDATES)
    order_item_name_col = find_column(order_df.columns, ORDER_ITEM_NAME_CANDIDATES)
    order_quantity_col = find_column(order_df.columns, ORDER_QUANTITY_CANDIDATES)
    order_created_at_col = find_column(order_df.columns, ORDER_CREATED_AT_CANDIDATES)

    cleaned = order_df.rename(
        columns={
            order_id_col: "order_id",
            order_item_code_col: "item_code",
            order_item_name_col: "item_name_raw",
            order_quantity_col: "quantity",
            order_created_at_col: "created_at",
        }
    ).copy()

    cleaned["order_id"] = cleaned["order_id"].map(normalize_lookup_text)
    cleaned["item_code"] = cleaned["item_code"].map(normalize_item_code)
    cleaned["item_name_raw"] = cleaned["item_name_raw"].map(normalize_lookup_text)
    cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce")
    cleaned["created_at"] = cleaned["created_at"].map(normalize_lookup_text)
    cleaned["_created_timestamp"] = pd.to_datetime(
        cleaned["created_at"],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )

    cleaned = cleaned.dropna(subset=["quantity", "_created_timestamp"]).copy()
    cleaned = cleaned[
        (cleaned["item_code"] != "")
        & (cleaned["order_id"] != "")
        & cleaned["quantity"].gt(0)
    ].copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    order_frequency = cleaned["item_code"].value_counts().astype(int).to_dict()
    order_name_lookup = (
        cleaned.assign(_name_key=cleaned["item_name_raw"].map(normalize_similarity_text))
        .groupby("item_code")["_name_key"]
        .apply(lambda series: [value for value in series.dropna().unique().tolist() if value])
        .to_dict()
    )
    return cleaned, order_frequency, order_name_lookup


def choose_best_candidate_item_code(
    title_value: object,
    query_value: object,
    candidates: pd.DataFrame,
    order_frequency: dict[str, int],
    order_name_lookup: dict[str, list[str]],
) -> tuple[str, str]:
    title_key = normalize_similarity_text(title_value)
    query_key = normalize_similarity_text(query_value)
    best_score: tuple[float, float, int, int, str] | None = None
    best_item_code = ""

    for candidate in candidates.to_dict("records"):
        item_code = candidate["item_code"]
        candidate_order_names = order_name_lookup.get(item_code, [])
        best_name_similarity = 0.0
        if candidate_order_names:
            best_name_similarity = max(
                max(similarity_score(title_key, name), similarity_score(query_key, name))
                for name in candidate_order_names
            )

        candidate_score = (
            float(item_code in order_frequency),
            best_name_similarity,
            int(order_frequency.get(item_code, 0)),
            int(candidate["_dimension_completeness"]),
            item_code,
        )
        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_item_code = item_code

    if best_score is None:
        return "", "no_candidate"
    if best_score[0] > 0:
        return best_item_code, "ambiguous_master_match_resolved_with_order_history"
    return best_item_code, "ambiguous_master_match_resolved_without_order_history"


def resolve_product_item_codes(
    product_df: pd.DataFrame,
    master_note_groups: dict[str, pd.DataFrame],
    order_frequency: dict[str, int],
    order_name_lookup: dict[str, list[str]],
) -> pd.DataFrame:
    resolved = product_df.copy()
    resolved["query"] = resolved["query"].map(normalize_lookup_text)
    resolved["title"] = resolved["title"].map(normalize_lookup_text)
    resolved["brand"] = resolved["brand"].map(normalize_lookup_text)
    resolved["category"] = resolved["category"].map(normalize_lookup_text)
    resolved["specification"] = resolved["specification"].map(normalize_lookup_text)
    resolved["capacity"] = resolved["capacity"].map(normalize_lookup_text)
    resolved["shelf_life"] = resolved["shelf_life"].map(normalize_lookup_text)
    resolved["Discount"] = resolved["Discount"].map(normalize_lookup_text)
    resolved["Promo"] = resolved["Promo"].map(normalize_lookup_text)

    item_codes: list[str] = []
    resolution_statuses: list[str] = []
    candidate_counts: list[int] = []
    order_supported_candidate_counts: list[int] = []

    for row in resolved.itertuples(index=False):
        candidates = master_note_groups.get(row.query)
        if candidates is None or candidates.empty:
            item_codes.append("")
            resolution_statuses.append("no_master_match")
            candidate_counts.append(0)
            order_supported_candidate_counts.append(0)
            continue

        candidate_count = len(candidates)
        supported_count = int(candidates["item_code"].isin(order_frequency).sum())
        candidate_counts.append(candidate_count)
        order_supported_candidate_counts.append(supported_count)

        if candidate_count == 1:
            item_codes.append(candidates.iloc[0]["item_code"])
            resolution_statuses.append("unique_master_match")
            continue

        chosen_item_code, resolution_status = choose_best_candidate_item_code(
            title_value=row.title,
            query_value=row.query,
            candidates=candidates,
            order_frequency=order_frequency,
            order_name_lookup=order_name_lookup,
        )
        item_codes.append(chosen_item_code)
        resolution_statuses.append(resolution_status)

    resolved["item_code"] = item_codes
    resolved["item_code_resolution_status"] = resolution_statuses
    resolved["master_candidate_count"] = candidate_counts
    resolved["order_supported_master_candidates"] = order_supported_candidate_counts
    return resolved


def extract_pieces_per_package(specification: object) -> int | None:
    text = normalize_lookup_text(specification).lower()
    if not text:
        return None

    x_numbers = re.findall(r"x\s*(\d+)", text)
    for value in x_numbers:
        if value != "1":
            return int(value)
    if x_numbers:
        return 1

    trailing_match = re.search(rf"(\d+)\s*(?:{COUNT_UNIT_PATTERN})\s*$", text)
    if trailing_match:
        return int(trailing_match.group(1))
    return None


def extract_package_unit(specification: object) -> str:
    text = normalize_lookup_text(specification).lower()
    if not text:
        return ""

    english_matches = re.findall(r"x\s*\d+\s*(?!x)([a-zA-Z]+)(?=[\u4e00-\u9fff]*$|$)", text)
    if english_matches:
        return UNIT_SYNONYMS.get(english_matches[-1], english_matches[-1])

    chinese_matches = re.findall(rf"x\s*\d+\s*(?!x)({COUNT_UNIT_PATTERN})", text)
    if chinese_matches:
        return COUNT_UNIT_MAP[chinese_matches[-1]]

    trailing_match = re.search(rf"(\d+)\s*({COUNT_UNIT_PATTERN})\s*$", text)
    if trailing_match:
        return COUNT_UNIT_MAP[trailing_match.group(2)]
    return ""


def extract_packaging_columns(df: pd.DataFrame) -> pd.DataFrame:
    extracted = df.copy()
    specification_normalized = extracted["specification"].astype(str).str.strip().str.lower()
    capacity_normalized = extracted["capacity"].astype(str).str.strip().str.lower()

    capacity_from_capacity = capacity_normalized.str.extract(
        r"(\d+(?:\.\d+)?)",
        expand=False,
    )
    capacity_unit_from_capacity = capacity_normalized.str.extract(
        r"\d+(?:\.\d+)?\s*([a-zA-Z]+)",
        expand=False,
    )

    capacity_from_specification = specification_normalized.str.extract(
        r"[:：]\s*(\d+(?:\.\d+)?)",
        expand=False,
    )
    capacity_unit_from_specification = specification_normalized.str.extract(
        r"[:：]\s*\d+(?:\.\d+)?\s*([a-zA-Z]+)",
        expand=False,
    )
    chinese_unit_from_specification = specification_normalized.str.extract(
        rf"[:：]?\s*\d+(?:\.\d+)?\s*({COUNT_UNIT_PATTERN})",
        expand=False,
    )

    extracted["capacity"] = capacity_from_capacity.where(
        capacity_unit_from_capacity.notna(),
        capacity_from_specification,
    ).fillna(capacity_from_capacity).fillna(capacity_from_specification)
    extracted["capacity_unit"] = capacity_unit_from_capacity.fillna(
        capacity_unit_from_specification
    )
    extracted["capacity_unit"] = extracted["capacity_unit"].fillna(
        chinese_unit_from_specification.map(COUNT_UNIT_MAP)
    )

    extracted["pieces_per_package"] = specification_normalized.map(extract_pieces_per_package).astype("Int64")
    extracted["package_unit"] = specification_normalized.map(extract_package_unit)
    return extracted


def normalize_original_price(value: object) -> float:
    if pd.isna(value):
        return np.nan

    raw_value = normalize_lookup_text(value)
    if not raw_value:
        return np.nan

    if re.fullmatch(r"\d+\.0+", raw_value):
        return float(raw_value.split(".")[0])
    if re.fullmatch(r"\d+\.\d{2}", raw_value):
        left, right = raw_value.split(".")
        return float(left + right + "0")
    if re.fullmatch(r"\d+\.\d{3}", raw_value):
        return float(raw_value.replace(".", ""))

    return pd.to_numeric(raw_value, errors="coerce")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    handled = df.copy()
    handled["pieces_per_package"] = handled["pieces_per_package"].fillna(1)
    handled["capacity_unit"] = handled["capacity_unit"].fillna("pc")
    handled["package_unit"] = handled["package_unit"].fillna("")
    handled["shelf_life"] = handled["shelf_life"].replace("", np.nan)
    handled["original_price"] = handled["original_price"].apply(normalize_original_price).fillna(0)
    return handled


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    priced = df.copy()
    priced["price"] = pd.to_numeric(priced["price"], errors="coerce")
    priced["pieces_per_package"] = pd.to_numeric(
        priced["pieces_per_package"],
        errors="coerce",
    ).fillna(1)
    priced["original_price"] = pd.to_numeric(priced["original_price"], errors="coerce").fillna(0)

    priced["price_per_piece"] = priced["price"] / priced["pieces_per_package"].replace(0, np.nan)
    priced["original_price_per_piece"] = (
        priced["original_price"] / priced["pieces_per_package"].replace(0, np.nan)
    )
    denominator = priced["original_price_per_piece"].replace(0, np.nan)
    priced["estimation_discount"] = (
        (priced["original_price_per_piece"] - priced["price_per_piece"]) / denominator * 100
    )
    priced["estimation_discount"] = (
        priced["estimation_discount"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0, upper=100)
    )
    return priced


def standardize_capacity(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.copy()
    standardized["capacity"] = pd.to_numeric(standardized["capacity"], errors="coerce")
    standardized["capacity_unit"] = (
        standardized["capacity_unit"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(UNIT_SYNONYMS)
    )

    unit_conversion_map = {
        "l": ("ml", 1000.0),
        "kg": ("g", 1000.0),
        "gal": ("ml", 3785.41),
        "lb": ("g", 453.592),
        "oz": ("ml", 29.5735),
        "m": ("cm", 100.0),
    }
    for source_unit, (target_unit, multiplier) in unit_conversion_map.items():
        mask = standardized["capacity_unit"].eq(source_unit)
        standardized.loc[mask, "capacity"] = standardized.loc[mask, "capacity"] * multiplier
        standardized.loc[mask, "capacity_unit"] = target_unit

    unit_category_map = {
        "ml": "volume",
        "cc": "volume",
        "g": "weight",
        "cm": "length",
        "pc": "count",
        "sheet": "count",
        "pack": "count",
        "bag": "count",
        "box": "count",
        "set": "count",
        "bottle": "count",
        "can": "count",
        "roll": "count",
        "pair": "count",
        "card": "count",
        "bucket": "count",
        "bowl": "count",
    }
    standardized["capacity_category"] = standardized["capacity_unit"].map(unit_category_map)
    return standardized


def normalize_shelf_life(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    raw_shelf_life = normalized["shelf_life"].astype(str).str.strip()

    extracted_value = raw_shelf_life.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    extracted_unit = raw_shelf_life.str.extract(r"([^\d.]+)", expand=False).fillna("")
    shelf_life_value = pd.to_numeric(extracted_value, errors="coerce")

    day_mask = extracted_unit.isin(["天", "日"])
    year_mask = extracted_unit.eq("年")

    shelf_life_value = shelf_life_value.where(~day_mask, shelf_life_value / 30.0)
    shelf_life_value = shelf_life_value.where(~year_mask, shelf_life_value * 12.0)
    shelf_life_value = shelf_life_value.fillna(999).clip(upper=999)

    normalized["shelf_life"] = np.ceil(shelf_life_value)
    normalized["shelf_life_unit"] = "month"
    return normalized


def add_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    modeled = df.copy()
    modeled["discount_numeric"] = pd.to_numeric(
        modeled["Discount"].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    ).fillna(0)
    modeled["promo_numeric"] = pd.to_numeric(
        modeled["Promo"],
        errors="coerce",
    ).fillna(0)
    return modeled


def deduplicate_product_item_codes(df: pd.DataFrame) -> pd.DataFrame:
    deduplicated = df.copy()
    deduplicated["_row_completeness"] = deduplicated[
        [
            "brand",
            "price",
            "original_price",
            "capacity",
            "shelf_life",
            "carton_length_cm",
            "carton_width_cm",
            "carton_height_cm",
            "carton_weight",
            "units_per_carton",
        ]
    ].notna().sum(axis=1)
    deduplicated = deduplicated.sort_values(
        ["_row_completeness", "master_candidate_count", "query"],
        ascending=[False, False, True],
    )
    deduplicated = deduplicated.drop_duplicates(subset=["item_code"], keep="first").copy()
    return deduplicated.drop(columns=["_row_completeness"])


def resolve_cross_dataset_conflicts(
    product_df: pd.DataFrame,
    order_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared_item_codes = sorted(set(product_df["item_code"]) & set(order_df["item_code"]))

    matched_product = product_df[product_df["item_code"].isin(shared_item_codes)].copy()
    matched_order = order_df[order_df["item_code"].isin(shared_item_codes)].copy()

    price_lookup = matched_product.drop_duplicates("item_code").set_index("item_code")["price_per_piece"]
    canonical_name_lookup = matched_product.drop_duplicates("item_code").set_index("item_code")["query"]

    matched_order["item_name"] = matched_order["item_code"].map(canonical_name_lookup).fillna(
        matched_order["item_name_raw"]
    )
    matched_order["price_per_piece"] = matched_order["item_code"].map(price_lookup)
    matched_order["sales"] = (matched_order["quantity"] * matched_order["price_per_piece"]).round(6)

    matched_order = matched_order[
        [
            "order_id",
            "item_code",
            "item_name",
            "quantity",
            "created_at",
            "price_per_piece",
            "sales",
        ]
    ].copy()

    matched_product = matched_product.sort_values(["item_code"]).reset_index(drop=True)
    matched_order = matched_order.sort_values(
        ["created_at", "order_id", "item_code", "item_name"]
    ).reset_index(drop=True)
    return matched_product, matched_order


def finalize_product_columns(df: pd.DataFrame) -> pd.DataFrame:
    final_columns = [
        "title",
        "category",
        "price",
        "brand",
        "specification",
        "query",
        "original_price",
        "original_price_checked",
        "capacity",
        "shelf_life",
        "order",
        "Discount",
        "Promo",
        "item_code",
        "item_code_resolution_status",
        "master_candidate_count",
        "order_supported_master_candidates",
        "capacity_unit",
        "pieces_per_package",
        "package_unit",
        "price_per_piece",
        "original_price_per_piece",
        "estimation_discount",
        "capacity_category",
        "shelf_life_unit",
        "discount_numeric",
        "promo_numeric",
        "carton_length_cm",
        "carton_width_cm",
        "carton_height_cm",
        "carton_weight",
        "units_per_carton",
    ]
    return df[final_columns].copy()


def build_preprocessed_datasets(
    raw_product_df: pd.DataFrame,
    master_product_df: pd.DataFrame,
    raw_order_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master_by_item_code, master_note_groups = prepare_master_product_data(master_product_df)
    cleaned_order_df, order_frequency, order_name_lookup = clean_order_dataset(raw_order_df)

    resolved_product_df = resolve_product_item_codes(
        product_df=raw_product_df,
        master_note_groups=master_note_groups,
        order_frequency=order_frequency,
        order_name_lookup=order_name_lookup,
    )
    resolved_product_df = resolved_product_df[resolved_product_df["item_code"] != ""].copy()
    resolved_product_df = resolved_product_df.merge(
        master_by_item_code[
            [
                "item_code",
                "carton_length_cm",
                "carton_width_cm",
                "carton_height_cm",
                "carton_weight",
                "units_per_carton",
            ]
        ],
        on="item_code",
        how="left",
    )

    resolved_product_df = extract_packaging_columns(resolved_product_df)
    resolved_product_df = handle_missing_values(resolved_product_df)
    resolved_product_df = add_price_features(resolved_product_df)
    resolved_product_df = standardize_capacity(resolved_product_df)
    resolved_product_df = normalize_shelf_life(resolved_product_df)
    resolved_product_df = add_modeling_columns(resolved_product_df)
    resolved_product_df = deduplicate_product_item_codes(resolved_product_df)

    aligned_product_df, aligned_order_df = resolve_cross_dataset_conflicts(
        product_df=resolved_product_df,
        order_df=cleaned_order_df,
    )

    return finalize_product_columns(aligned_product_df), aligned_order_df


def build_verification_summary(
    product_df: pd.DataFrame,
    order_df: pd.DataFrame,
) -> pd.DataFrame:
    product_codes = set(product_df["item_code"].map(normalize_item_code))
    order_codes = set(order_df["item_code"].map(normalize_item_code))

    verification_records = [
        {
            "section": "verification",
            "dataset": "product_final",
            "column": "",
            "metric": "row_count",
            "value": len(product_df),
        },
        {
            "section": "verification",
            "dataset": "order_final",
            "column": "",
            "metric": "row_count",
            "value": len(order_df),
        },
        {
            "section": "verification",
            "dataset": "alignment",
            "column": "",
            "metric": "product_unique_item_codes",
            "value": len(product_codes),
        },
        {
            "section": "verification",
            "dataset": "alignment",
            "column": "",
            "metric": "order_unique_item_codes",
            "value": len(order_codes),
        },
        {
            "section": "verification",
            "dataset": "alignment",
            "column": "",
            "metric": "product_only_item_codes",
            "value": len(product_codes - order_codes),
        },
        {
            "section": "verification",
            "dataset": "alignment",
            "column": "",
            "metric": "order_only_item_codes",
            "value": len(order_codes - product_codes),
        },
        {
            "section": "verification",
            "dataset": "product_final",
            "column": "item_code",
            "metric": "missing_item_code_rows",
            "value": int(product_df["item_code"].eq("").sum()),
        },
        {
            "section": "verification",
            "dataset": "product_final",
            "column": "item_code",
            "metric": "duplicate_item_code_rows",
            "value": int(product_df.duplicated(subset=["item_code"]).sum()),
        },
        {
            "section": "verification",
            "dataset": "order_final",
            "column": "item_code",
            "metric": "missing_item_code_rows",
            "value": int(order_df["item_code"].eq("").sum()),
        },
        {
            "section": "verification",
            "dataset": "order_final",
            "column": "",
            "metric": "exact_duplicate_rows",
            "value": int(order_df.duplicated().sum()),
        },
        {
            "section": "verification",
            "dataset": "order_final",
            "column": "created_at",
            "metric": "invalid_timestamps",
            "value": int(
                pd.to_datetime(
                    order_df["created_at"],
                    format="%d/%m/%Y %H:%M",
                    errors="coerce",
                ).isna().sum()
            ),
        },
        {
            "section": "verification",
            "dataset": "order_final",
            "column": "quantity",
            "metric": "nonpositive_quantities",
            "value": int(pd.to_numeric(order_df["quantity"], errors="coerce").le(0).fillna(True).sum()),
        },
    ]
    return pd.DataFrame(verification_records)


def export_filtered_frames(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for category, capacity_categories in FILTER_EXPORTS:
        for capacity_category in capacity_categories:
            subset = df[
                (df["category"] == category)
                & (df["capacity_category"] == capacity_category)
            ].copy()
            output_path = output_dir / f"df_{category}_{capacity_category}.csv"
            subset.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
                sep=";",
                decimal=",",
            )
            output_paths.append(output_path)

    return output_paths


def main() -> None:
    raw_product_path = find_existing_path(
        RAW_PRODUCT_DATA_CANDIDATES,
        "webscraping_result.csv",
    )
    master_product_path = find_existing_path(
        MASTER_PRODUCT_DATA_CANDIDATES,
        "儲格設計_原檔(商品資訊).csv",
    )
    raw_order_path = find_existing_path(
        RAW_ORDER_DATA_CANDIDATES,
        "訂單資料(order data).csv",
    )

    raw_product_df = load_raw_product_data(raw_product_path)
    master_product_df = load_master_product_data(master_product_path)
    raw_order_df = load_raw_order_data(raw_order_path)

    analysis_summary_df, issue_counts = analyze_raw_datasets(
        product_df=raw_product_df,
        master_df=master_product_df,
        order_df=raw_order_df,
    )
    rule_summary_df = build_rule_summary(issue_counts)

    product_final_df, order_final_df = build_preprocessed_datasets(
        raw_product_df=raw_product_df,
        master_product_df=master_product_df,
        raw_order_df=raw_order_df,
    )
    verification_summary_df = build_verification_summary(
        product_df=product_final_df,
        order_df=order_final_df,
    )

    PRODUCT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    product_output_path = save_dataframe(
        product_final_df,
        PRODUCT_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )
    order_output_path = save_dataframe(
        order_final_df,
        ORDER_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        sep=";",
        decimal=",",
    )
    analysis_output_path = save_dataframe(
        pd.concat([analysis_summary_df, rule_summary_df], ignore_index=True),
        ANALYSIS_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    verification_output_path = save_dataframe(
        verification_summary_df,
        VERIFICATION_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    filtered_paths = export_filtered_frames(product_final_df, FILTERED_DATA_DIR)

    print(f"Raw product data:      {safe_console_text(raw_product_path)}")
    print(f"Master product data:   {safe_console_text(master_product_path)}")
    print(f"Raw order data:        {safe_console_text(raw_order_path)}")
    print(f"Product rows exported: {len(product_final_df):,}")
    print(f"Order rows exported:   {len(order_final_df):,}")
    print(f"Product output CSV:    {safe_console_text(product_output_path)}")
    print(f"Order output CSV:      {safe_console_text(order_output_path)}")
    print(f"Analysis summary:      {safe_console_text(analysis_output_path)}")
    print(f"Verification summary:  {safe_console_text(verification_output_path)}")
    print(
        "Filtered exports:      "
        f"{len(filtered_paths)} files in {safe_console_text(FILTERED_DATA_DIR)}"
    )


if __name__ == "__main__":
    main()
