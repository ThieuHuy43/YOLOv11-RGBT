import warnings
warnings.filterwarnings('ignore')
import comet_ml

from ultralytics import YOLO

def main():
    comet_ml.login(api_key="xdC9oTmhNXcGwJxE0iAerSNG1")
    # Load mô hình YOLOv11-RGBT với late fusion
    model = YOLO("ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml")
    
    # Huấn luyện mô hình
    model.train(
        data=r'C:\Users\PC\OneDrive\Dokumen\DL4CV\DL2\YOLOv11-RGBT\data_rgbt.yaml',  # Tệp data.yaml chỉ định visible, thermal, labels
        imgsz=640,
        epochs=2000,
        batch=0.99,
        device='0',             # GPU 0
        use_simotm="RGBT",      # Kích hoạt chế độ RGBT late fusion
        channels=4,             # RGB (3) + Thermal (1), tổng cộng 4 kênh
        project='runs/train',   # Thư mục chứa kết quả
        name='yolov11-rgbt',    # Tên experiment
        workers=6,              # Số lượng luồng xử lý dữ liệu
        patience = 15,
        save = True,
        verbose=True            # Hiển thị chi tiết quá trình huấn luyện
    )

if __name__ == '__main__':
    main()