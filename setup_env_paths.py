"""
Legacy path-helper compatibility script.

Các file YAML/JSON/split của pipeline hiện dùng đường dẫn tuyệt đối được sinh
tại runtime bởi `pcb_pipeline/data_module.py`. Script này KHÔNG còn sửa đường
dẫn nữa (sửa thủ công dễ làm sai split hoặc làm lẫn official val/test).

Chức năng hiện tại: kiểm tra nhanh data prep đã hợp lệ hay chưa.
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kiểm tra data prep (không sửa đường dẫn)"
    )
    parser.add_argument(
        "--mode",
        default="relative",
        choices=["relative", "absolute"],
        help="Tham số cũ, được giữ để không làm hỏng script cũ; hiện bị bỏ qua.",
    )
    args = parser.parse_args()

    from run_pipeline import DATA_DIR, _validate_data_prep

    ok, reason = _validate_data_prep(DATA_DIR, 799)
    print("=" * 66)
    print("KIỂM TRA DATA PREP")
    print("=" * 66)
    print(f"Thư mục dữ liệu: {DATA_DIR}")
    print(f"Trạng thái:      {'HỢP LỆ' if ok else 'KHÔNG HỢP LỆ'}")
    print(f"Chi tiết:        {reason}")
    print("Lưu ý: script không sửa file nào; --mode chỉ để tương thích lệnh cũ.")
    print("=" * 66)

    if not ok:
        print("Hãy chạy: python run_pipeline.py prepare_data --force")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
