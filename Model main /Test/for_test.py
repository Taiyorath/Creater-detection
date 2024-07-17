from ultralytics import YOLO

model = YOLO('tes/best.pt') #path of the best weight 

results = model.predict('tes/lots-cars-on-city-road-view-from-above-E5DY28.jpg', imgsz=640, conf=0.3, save=True, show=True)
#specify the path of the image to be tested