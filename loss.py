import tensorflow as tf
import numpy as np
from utils import iou # util.py에서 IOU를 계산해주는 함수 가져오기.

def yolo_loss(predict,
              labels,
              each_object_num,
              num_classes,
              boxes_per_cell,
              cell_size,
              input_width,
              input_height,
              coord_scale,
              object_scale,
              noobject_scale,
              class_scale
              ):
  '''
          파라미터 설명
          predict: [S, S, B * 5 + C]
          labels: 정답 object 갯수 만큼의 (x, y, w, h, c) 벡터 [갯수, 5] x, y, w,h -> 모두 절대경로.
          each_object_num: train.py에서 호출할 때 몇번째 오브젝트에 대한 index integer.
          num_classes: 클래스 갯수, C
          boxes_per_cell: 한개의 grid cell에 몇개의 박스가 존재할지, B
          cell_size: 전체 이미지를 몇개의 grid cell로 나눌건지, S
          input_width: 원본 이미지의 w
          input_height: 원본 이미지의 h
          coord_scale: coordinate loss에 적용될 가중치
          object_scale: object loss에 적용될 가중치
          noobject_scale: object가 존재하지 않을 경우 object loss에 적용될 가중치
          class_scale: class loss에 적용될 가중치.



  '''
 

  # predict: [S, S, B * 5 + C], 앞 20개는 class값, 그 후 n개는 boxes per cell. 나머지가 이제 xy, y, w, h per box
  # 좌표값 벡터에 해당하는 값들만 파싱.
  predict_boxes = predict[:, :, num_classes + boxes_per_cell:]
  predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4]) # 4: coordinate vectors

  # 실제 이미지에 대한 절대 좌표값 계산.
  pred_xcenter = predict_boxes[:, :, :, 0]
  pred_ycenter = predict_boxes[:, :, :, 1]
  pred_sqrt_w = tf.sqrt(tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2]))) # 예측값 자체가 이미지의 최대 최소 width 사이의 값을 가지도록 restrict.
  pred_sqrt_h = tf.sqrt(tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3]))) # 예측값 자체가 이미지의 최대 최소 height 사이의 값을 가지도록 restrict.
  pred_sqrt_w = tf.cast(pred_sqrt_w, tf.float32) # float32 값으로 type casting
  pred_sqrt_h = tf.cast(pred_sqrt_h, tf.float32) # float32 값으로 type casting

  # labels: 정답 object 갯수 만큼의 (x, y, w, h, c) 벡터 [갯수, 5] x, y, w,h -> 모두 절대경로.
  # 정답값에 해당하는 라벨을 파싱.
  labels = np.array(labels)
  labels = labels.astype('float32')
  label = labels[each_object_num, :]
  xcenter = label[0]
  ycenter = label[1]
  sqrt_w = tf.sqrt(label[2])
  sqrt_h = tf.sqrt(label[3])

  # Ground Truth 값과 예측 결과의 차이를 계산.
  iou_predict_truth = iou(predict_boxes, label[0:4])

  # n개의 Bounding Box 중에서 가장 높은 IOU 값을 가지고 있는 박스에는 1로, 아니면 0으로 마스킹 처리.
  I = iou_predict_truth # I : [C, C, B] e.g. (7, 7, 2)
  max_I = tf.reduce_max(I, 2, keepdims=True) # 2개의 bounding box 중에서 가장 큰 IOU 값을 가지고 있는 박스를 추출. e.g. (7, 7, 1)
  best_box_mask = tf.cast((I >= max_I), tf.float32) # max_I에 해당하는 박스면 1.0, 아니면 0.0으로 처리. e.g. (7, 7, 2)

  # n개의 Bounding Box별로 존재하는 confidence에 따른 오차 계산을 위한 준비.
  C = iou_predict_truth # I : [C, C, B] e.g. (7, 7, 2)
  pred_C = predict[:, :, num_classes : num_classes + boxes_per_cell]

  # 20개의 class별 softmax regression 값 준비.
  P = tf.one_hot(tf.cast(label[4], tf.int32), num_classes, dtype=tf.float32)
  pred_P = predict[:, :, 0:num_classes]

  # find object exists cell mask
  object_exists_cell = np.zeros([cell_size, cell_size, 1]) # 전체 cell_size 만큼의 벡터값을 가지고 있다가
  # xcenter, ycenter가 정답 object가 존재하는 위치라는 점을 활용한다.
  # (ycenter / input_height), (xcenter / input_width)는 각각 이미지 사이즈에 맞게 normalised된 이미지의 중앙 위치이기 때문에,
  # cell size와 곱하고 ceil을 적용하여 몇번째 박스에 이미지가 실제로 존재한다고 할 수 있다.
  object_exists_cell_i, object_exists_cell_j = int(cell_size * ycenter / input_height), int(cell_size * xcenter / input_width)
  object_exists_cell[object_exists_cell_i][object_exists_cell_j] = 1 # 해당 cell에 이미지가 존재할 경우 1로 마스킹 해준다.

  # 좌표값에 대한 손실값 구하기.
  # 현재 pred_(x,y)center, (x,y)center는 모두 절대 좌표로 convert 되어 있기 때문에, 논문에서 언급한것 처럼 cell 내의 상대 좌표로 아래와 같이 convert.
  # w, h 값 역시 절대 좌표로 되어있기 때문에, input (w,h)로 나눠주면서 전체 이미지 대비 scale로 normalise 진행.
  coord_loss = (tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_xcenter - xcenter) / (input_width / cell_size)) +
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_ycenter - ycenter) / (input_height / cell_size)) +
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_w - sqrt_w)) / input_width +
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_h - sqrt_h)) / input_height ) \
               * coord_scale

  # Object가 존재하는 경우에 대한 confidence prediction 손실값 계산.
  object_loss = tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_C - C)) * object_scale

  # Object가 존재하지 않는 경우에 대한 confidence prediction 손실값 계산.
  noobject_loss = tf.nn.l2_loss((1 - object_exists_cell) * (pred_C)) * noobject_scale

  # classification 손실값 계산.
  class_loss = tf.nn.l2_loss(object_exists_cell * (pred_P - P)) * class_scale

  # 모든 손실값의 합 계산.
  total_loss = coord_loss + object_loss + noobject_loss + class_loss

  # tensorboard와 같이 현재 진행중인 훈련에서 어떠한 loss 값에 대한 개선이 이루어지고 있는지, 또 이루어지고 있지 않는지 확인하기 위해 모든 손실값을 반환.
  return total_loss, coord_loss, object_loss, noobject_loss, class_loss
