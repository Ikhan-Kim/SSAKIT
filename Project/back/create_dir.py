import os

os.makedirs("./dataset/train")
os.makedirs("./dataset/validation")
os.makedirs("./dataset/test")

 
# mkdir은 한 폴더만 생성 가능하며 하위폴더 생성이 불가함
# makedirs는 './a/b/c'처럼 원하는만큼 디렉토리 생성이 가능
# exist_ok 파라미터를 True로 설정하면 디렉토리가 기존에 존재해도
# 에러발생 없이 넘어가고, 없을 경우에만 디렉토리를 생성합니다.
# 반대로 exist_ok를 True로 설정하지 않았을 때 이미 해당 디렉토리가 
# 존재하는 경우에는 exception 에러가 뜨게 된다.
# os.makedirs("./dataset/train, exist_ok=True")