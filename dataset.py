import tensorflow as tf
import numpy as np

# Reference : https://stackoverflow.com/questions/54567986/python-numpy-remove-empty-zeroes-border-of-3d-array
def bounds_per_dimension(ndarray):
  return map(
    lambda e: range(e.min(), e.max() + 1),
    np.where(ndarray != 0) # 0에 해당하지 않는 부분만 넘겨준다.
  )

# 이미지를 배치별로 받아오기 때문에, 배치 안에서 이미지 사이즈가 상이할 경우, 사이즈가 더 작은 이미지는 해당 배치 안에서 사이즈가 제일 큰 이미지만큼의 공간에
# zero padding을 적용한 상태로 넘어오게 된다.
def zero_trim_ndarray(ndarray):
  return ndarray[np.ix_(*bounds_per_dimension(ndarray))]


# YOLO 포멧에 맞게 Ground Truth 데이터를 전처리.
def process_each_ground_truth(original_image,
                              bbox,
                              class_labels,
                              input_width,
                              input_height
                              ):
  '''
          파라미터 설명
          original_image: 논문상에선 이미지를 484, 484로 resizing 하는데, resizing 하기 전 원본 이미지의 [height, width, channel] 텐서.
          bbox: PASCAL VOC dataset 의 정답 G.T data가 (ymin / height, xmin / width, ymax / height, xmax / width) 포맷으로 저장되어 있다.
                (max_object_num_in_batch, 4)
          class_labels: OHE 처리가 되어있지 않은 정답 라벨링.  (max_object_num_in_batch)
          input_width: resizing 해서 넣는 yolo input width
          input_height: resizing 해서 넣는 yolo input height
  '''


  image = original_image.numpy() # 텐서 타입으로 받아 온 이미지를 넘파이 포맷으로 변경.
  image = zero_trim_ndarray(image) # 이미지에서 0인 부분을 전부 다 잘라주기.

  # 원본 이미지의 width, height 값 가져오기.
  original_h = image.shape[0]
  original_w = image.shape[1]

  # resize 된 이미지 대비 원 이미지의 width, height 비율 구하기.
  width_rate = input_width * 1.0 / original_w
  height_rate = input_height * 1.0 / original_h

  image = tf.image.resize(image, [input_height, input_width]) # (resized_height, resized_width, channel) image ndarray

  # object_num: 이미지에 실제로 존재하는 모든 객채들의 갯수.
  object_num = np.count_nonzero(bbox, axis=0)[0] # 전체 바운딩 박스 안에서 0이 포함되지 않고 실제로 object가 있는 갯수.
  # 이 역시 배치 단위로 받아오는 과정에서 필요한 파싱인데, 배치 안에 3개의 이미지가 있다고 가정할 때 한 이미지가 [1, 1, 1] 이렇게 있다면 나머지 이미지들은
  # 객체가 하나밖에 없어도 [1, 0, 0]이런 식으로 패딩이 적용되어 있기 때문에 위와 같은 파싱 과정이 필요하다.

  # 2-D list [object_num, 5] (xcenter (절대경로), ycenter (절대경로), w (절대경로), h (절대경), class_num)
  labels = [[0, 0, 0, 0, 0]] * object_num # 리턴값에 대한 초기값 initialising
  for i in range(object_num): # 오브젝트 갯수만큼 반복하며 (x,y)min, (x,y)max를 절대 좌표로 변경하는 과정.
    xmin = bbox[i][1] * original_w
    ymin = bbox[i][0] * original_h
    xmax = bbox[i][3] * original_w
    ymax = bbox[i][2] * original_h

    class_num = class_labels[i] # OHE 처리가 되어있지 않은 정답 라벨링.  (max_object_num_in_batch)

    xcenter = (xmin + xmax) * 1.0 / 2 * width_rate # 절대좌표로 변경.
    ycenter = (ymin + ymax) * 1.0 / 2 * height_rate # 절대좌표로 변경.

    box_w = (xmax - xmin) * width_rate # 절대좌표로 변경.
    box_h = (ymax - ymin) * height_rate # 절대좌표로 변경.

    labels[i] = [xcenter, ycenter, box_w, box_h, class_num]


  return [image.numpy(), labels, object_num]
