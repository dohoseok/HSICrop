wget -O dataset/download.zip https://kr.object.ncloudstorage.com/aidata-sample/20-02/2020-02-067.%EB%93%9C%EB%A1%A0%EB%86%8D%EA%B2%BD%EC%9E%91%EC%A7%80_sample.zip
unzip dataset/download.zip -d dataset/
mv dataset/[라벨]label dataset/label
mv dataset/[원천]data dataset/image
python build_np.py