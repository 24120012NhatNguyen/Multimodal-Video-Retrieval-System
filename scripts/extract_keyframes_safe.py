import os
import sys
import zipfile
import json
import zlib
import time

def main():
    zip_path = os.path.abspath("data/keyframe.zip")
    dest_dir = os.path.abspath("data")
    state_path = os.path.abspath("data/keyframe_extract_state.json")

    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} không tồn tại.", file=sys.stderr)
        sys.exit(1)

    # 1. Khởi tạo danh sách trạng thái ban đầu nếu chưa có
    if not os.path.exists(state_path):
        print("Đang đọc cấu trúc file zip lần đầu để lưu trạng thái...")
        t0 = time.time()
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                state = []
                for member in zf.infolist():
                    state.append({
                        "filename": member.filename,
                        "header_offset": member.header_offset,
                        "compress_size": member.compress_size,
                        "file_size": member.file_size,
                        "compress_type": member.compress_type
                    })
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            print(f"Đã lưu thông tin cấu trúc zip vào {state_path} trong {time.time() - t0:.2f}s.")
        except Exception as e:
            print(f"Lỗi khi đọc file zip: {e}", file=sys.stderr)
            sys.exit(1)

    # 2. Đọc trạng thái từ file JSON
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    # 3. Lấy dung lượng file zip hiện tại để biết tiến trình (hỗ trợ resume)
    current_zip_size = os.path.getsize(zip_path)
    total_original_size = max(m["header_offset"] + m["compress_size"] for m in state) if state else 1
    
    # Lọc những phần tử chưa được giải nén (có offset nằm trong khoảng file zip hiện tại)
    remaining = [m for m in state if m["header_offset"] < current_zip_size]
    # Sắp xếp giảm dần theo offset để giải nén từ cuối lên đầu
    remaining.sort(key=lambda x: x["header_offset"], reverse=True)

    total_count = len(state)
    remaining_count = len(remaining)
    extracted_count = total_count - remaining_count

    print(f"Tổng số phần tử: {total_count}")
    print(f"Đã giải nén trước đó: {extracted_count}")
    print(f"Còn lại cần giải nén: {remaining_count}")
    print(f"Dung lượng zip hiện tại: {current_zip_size / (1024**3):.2f} GB")

    if not remaining:
        print("Không còn phần tử nào cần giải nén.")
        cleanup(zip_path, state_path)
        return

    # Cache các thư mục đã tạo để tránh gọi os.makedirs liên tục làm chậm tốc độ
    created_dirs = set()
    cache_root = os.path.join(dest_dir, "keyframe_cache")
    if os.path.exists(cache_root):
        print("Đang quét các thư mục đã tồn tại...")
        for root, dirs, _ in os.walk(cache_root):
            created_dirs.add(root)

    t_start = time.time()
    last_log_time = t_start
    bytes_extracted = 0

    print("\nBắt đầu giải nén song song với xóa/truncating file zip...")
    
    try:
        for idx, m in enumerate(remaining):
            filename = m["filename"]
            header_offset = m["header_offset"]
            dest_filepath = os.path.join(dest_dir, filename)

            # Nếu là thư mục
            if filename.endswith("/"):
                if dest_filepath not in created_dirs:
                    os.makedirs(dest_filepath, exist_ok=True)
                    created_dirs.add(dest_filepath)
                # Truncate file zip
                with open(zip_path, "rb+") as wf:
                    wf.truncate(header_offset)
                continue

            # Tạo thư mục cha nếu chưa có
            parent_dir = os.path.dirname(dest_filepath)
            if parent_dir not in created_dirs:
                os.makedirs(parent_dir, exist_ok=True)
                created_dirs.add(parent_dir)

            # Đọc và giải nén dữ liệu
            with open(zip_path, "rb") as rf:
                rf.seek(header_offset)
                magic = rf.read(4)
                if magic != b"PK\x03\x04":
                    print(f"\nLỗi: Magic header sai tại offset {header_offset} cho file {filename}", file=sys.stderr)
                    sys.exit(1)
                
                # Đọc độ dài filename và extra field
                rf.seek(header_offset + 26)
                filename_len = int.from_bytes(rf.read(2), "little")
                extra_len = int.from_bytes(rf.read(2), "little")
                
                # Tìm offset của phần dữ liệu nén
                data_offset = header_offset + 30 + filename_len + extra_len
                rf.seek(data_offset)
                compressed_data = rf.read(m["compress_size"])

                # Decompress dữ liệu
                if m["compress_type"] == 8:  # Deflate
                    decompressed = zlib.decompress(compressed_data, -15)
                elif m["compress_type"] == 0:  # Stored
                    decompressed = compressed_data
                else:
                    print(f"\nLỗi: Kiểu nén không hỗ trợ {m['compress_type']} tại file {filename}", file=sys.stderr)
                    sys.exit(1)

                if len(decompressed) != m["file_size"]:
                    print(f"\nLỗi: Sai lệch kích thước giải nén cho file {filename}", file=sys.stderr)
                    sys.exit(1)

            # Ghi file tạm rồi rename để đảm bảo tính nguyên tử (atomic)
            tmp_filepath = dest_filepath + ".tmp"
            with open(tmp_filepath, "wb") as wf:
                wf.write(decompressed)
            os.replace(tmp_filepath, dest_filepath)

            # Truncate file zip ngay sau khi ghi thành công file ảnh
            with open(zip_path, "rb+") as wf:
                wf.truncate(header_offset)

            bytes_extracted += m["file_size"]
            extracted_count += 1

            # Log tiến trình mỗi 2 giây hoặc 1000 ảnh
            now = time.time()
            if now - last_log_time >= 2.0 or extracted_count % 1000 == 0 or idx == remaining_count - 1:
                elapsed = now - t_start
                speed = (bytes_extracted / (1024**2)) / elapsed if elapsed > 0 else 0
                pct_zip_remaining = (header_offset / total_original_size) * 100
                print(f"\rTiến trình: {extracted_count}/{total_count} files ({extracted_count/total_count*100:.1f}%) "
                      f"| File zip còn lại: {header_offset / (1024**3):.2f} GB ({pct_zip_remaining:.1f}%) "
                      f"| Tốc độ ghi: {speed:.2f} MB/s", end="", flush=True)
                last_log_time = now

        print("\n\nGiải nén hoàn tất thành công!")
        cleanup(zip_path, state_path)

    except KeyboardInterrupt:
        print("\n\nQuá trình bị tạm dừng bởi người dùng. Bạn có thể chạy lại script này bất cứ lúc nào để tiếp tục.")
    except Exception as e:
        print(f"\n\nGặp lỗi trong quá trình xử lý: {e}", file=sys.stderr)

def cleanup(zip_path, state_path):
    print("Đang dọn dẹp các file tạm...")
    if os.path.exists(zip_path):
        try:
            if os.path.getsize(zip_path) == 0:
                os.remove(zip_path)
                print("Đã xóa file keyframe.zip trống.")
        except Exception as e:
            print(f"Không thể xóa file zip: {e}", file=sys.stderr)
    if os.path.exists(state_path):
        try:
            os.remove(state_path)
            print("Đã xóa file trạng thái JSON.")
        except Exception as e:
            print(f"Không thể xóa file trạng thái: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
