import csv, json
import os
import io

kis_csv = '''sample_id,query_text,video_id,s,e,example_frame_id
sample_01,"Tìm video quay cận cảnh nhiều quả trứng màu trắng được xếp thành các hàng ngay ngắn trên nền cát đất ngoài trời. Và có thông tin nhắc đến nhóm ""Ngũ Bang""",L21_V003,7175,7275,7179
sample_04,"Tìm video quay cảnh người đàn ông mặc áo sơ mi xanh lá đeo kính ngồi trên xích lô ngoài đường phố, phía sau có một chiếc xe buýt màu cam vừa chạy vượt qua.",L27_V001,1750,1900,1889
sample_06,"Tìm video quay cảnh một bé trai mặc áo thun màu cam in hình đang thích thú cầm dây thả diều, bên cạnh có một phụ nữ mặc áo sơ mi và quần màu trắng chống tay vào hông nhìn theo.",L27_V010,12500,12550,12514
sample_08,"Tìm video quay cảnh ba người đàn ông đang ngồi quanh một bàn ăn trải khăn xanh trước hiên nhà, trong đó người đàn ông mặc áo thun xanh lá đeo kính bên trái đang dùng tay bốc thức ăn cho vào miệng.",L27_V016,11375,11425,11382
sample_10,Tìm video quay cận cảnh hai bàn tay đang cầm hai con cua rừng/cua núi nhỏ màu đỏ tím giơ lên trước ống kính ngoài tự nhiên.,L27_V004,11500,11625,11507
sample_11,"Tìm video ghi lại cảnh biểu diễn lân sư rồng ngoài trời, có con lân màu vàng đang đứng sau dàn cột Mai Hoa Thung, bên trái là mô hình lồng đèn/quả cầu màu đỏ vàng và góc dưới bên phải có một người nam mặc áo đen đang cầm máy ảnh chụp.",L24_V026,5725,5850,5828
sample_12,"Tìm video diễn án phiên tòa giả định tại trường SIU (Saigon International University), cảnh một người đàn ông áo đen đứng quay lưng đối diện hội đồng xét xử, bên trái có nữ thư ký mặc vest đen đang đứng cầm micro phát biểu.",L25_V013,1678,1708,1672
sample_13,"Tìm video quay cảnh các học sinh tiểu học mặc đồng phục xếp hàng lấy thức ăn buffet tại căng tin trường học tràn ngập ánh sáng, nổi bật với một bé gái thắt bím tóc đeo băng đô trắng đang tươi cười dùng kẹp gắp món ăn vào đĩa.",L25_V084,23556,23586,23620'''

qa_csv = '''sample_id,query_text,video_id,s,e,example_frame_id,answer
sample_02,"Trong video thời sự tại quầy giao dịch, cô gái buộc tóc đuôi ngựa, mặc áo khoác trắng có mũ, ngồi đối diện cô thu ngân mặc áo xanh lá, đang làm gì với chiếc điện thoại?",L21_V002,1560,1590,1565,"Quét mã QR"
sample_07,"Ở phần kết thúc chương trình 'Việt Nam Đi Là Ghiền' giữa cánh đồng khóm, người phụ nữ mặc áo hồng quàng khăn rằn nói về thông tin gì?",L27_V013,13325,13450,13500,"Thời gian phát sóng"
sample_09,"Trong video cảnh mua đồ ăn ở quầy hàng lúc trời mưa, có bao nhiêu bịch/bì nước lèo (nước bún) màu vàng cam được cột sẵn đặt ở mép bàn?",L27_V006,4150,4225,4165,"2"'''

out = []

f = io.StringIO(kis_csv.strip())
for r in csv.DictReader(f):
    if not r.get("sample_id"): continue
    out.append({
        "sample_id": r["sample_id"],
        "query_text": r["query_text"],
        "video_id": r["video_id"],
        "n_events": 1,
        "event1_s": int(r["s"]),
        "event1_e": int(r["e"])
    })

f = io.StringIO(qa_csv.strip())
for r in csv.DictReader(f):
    if not r.get("sample_id"): continue
    out.append({
        "sample_id": r["sample_id"],
        "query_text": r["query_text"],
        "video_id": r["video_id"],
        "n_events": 1,
        "event1_s": int(r["s"]),
        "event1_e": int(r["e"])
    })

if os.path.exists('eval/trake_ground_truth.json'):
    with open('eval/trake_ground_truth.json', encoding='utf-8') as f:
        out.extend(json.load(f))

out.sort(key=lambda x: x["sample_id"])

os.makedirs('data', exist_ok=True)
with open('data/synth_dev.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Done generating synth_dev.json with {len(out)} samples")
