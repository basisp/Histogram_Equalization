import cv2
import matplotlib.pyplot as plt

#히스토그램 평활화 함수
def histogram_equalization(m_OpenImg):
    hist = [0] * 256
    sum_of_hist = [0] * 256
    total_pixels = 256 * 256
    sum = 0

    for i in range(256):
        for j in range(256):
            k = int(m_OpenImg[i][j])
            hist[k] += 1

    for i in range(256):
        sum += hist[i]
        sum_of_hist[i] = sum

    m_ResultImg = [[0 for _ in range(256)] for _ in range(256)]
    for i in range(256):
        for j in range(256):
            k = int(m_OpenImg[i][j])  
            m_ResultImg[i][j] = int(sum_of_hist[k] * (255.0 / total_pixels))

    return m_ResultImg


def calculate_histogram(image):
    # 이미지 데이터의 범위가 0에서 255이므로, 256개의 공간을 갖는 리스트 생성
    histogram = [0 for _ in range(256)]
    # 이미지의 각 픽셀에 대해 히스토그램을 업데이트
    for pixel in [pixel for row in image for pixel in row]: #각 행에 있는 픽셀을 불러오는 반복문
        histogram[pixel] += 1
    return histogram


# CDF를 계산하는 함수
def calculate_cdf(histogram):
    total_pixels = sum(histogram)  # 전체 픽셀 수 계산
    pmf = [pixel_num / total_pixels for pixel_num in histogram]  # 픽셀의 개수/전체 픽셀 수
    cdf = []
    pmf_sum = 0
    for value in pmf:
        pmf_sum += value  # 누적 합산
        cdf.append(pmf_sum)  # CDF에 추가
    return cdf


#기본적으로 OpenCV는 이미지를 BGR(Blue, Green, Red) 형식으로 처리하지만, 
#matplotlib은 RGB(Red, Green, Blue) 형식으로 이미지를 처리한다.
#따라서 먼저 OpenCV로 불러온 이미지를 BGR -> YCbCr 로 변환하고 평활화한 후에 YCbCr->RGB 로 변환해 matplot으로 출력한다


def convert_to_YCbCr(BGR):
    YCbCr = []
    for row in BGR:
        YCbCr_row = []
        for pixel in row:
            B, G, R = pixel
            Y = 0.299*R + 0.587*G + 0.114*B
            Cb = (B - Y) * 0.564 + 128
            Cr = (R - Y) * 0.713 + 128
            YCbCr_row.append([Y, Cb, Cr])
        YCbCr.append(YCbCr_row)
    return YCbCr


def convert_to_RGB(YCbCr):
    RGB = []
    for row in YCbCr:  # YCbCr 이미지의 각 행에 대해 반복
        RGB_row = []
        for y, cb, cr in row:  # 각 행의 각 픽셀(Y, Cb, Cr)에 대해 반복
            # YCbCr를 RGB로 변환
            r = y + 1.402 * (cr - 128)
            g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
            b = y + 1.772 * (cb - 128)
            RGB_row.append([clip(int(r)), clip(int(g)), clip(int(b))]) #r,g,b가 정수가 아닐수도 있으므로 int 붙여줌
        RGB.append(RGB_row)  # 변환된 RGB 행을 최종 RGB 이미지에 추가
    return RGB


#값을 0과 255 사이로 제한하는 함수
def clip(value):
    return max(0, min(255, value))


#이미지의 R, G, B 채널을 분리해서 반환하는 함수
def split_channels(image):

    # 이미지의 높이와 너비를 구합니다.
    height = len(image)
    width = len(image[0]) if height > 0 else 0 #이미지가 비어있을 경우

    # R, G, B 채널을 저장할 2차원 리스트 초기화
    R_channel = [[0 for _ in range(width)] for _ in range(height)]
    G_channel = [[0 for _ in range(width)] for _ in range(height)]
    B_channel = [[0 for _ in range(width)] for _ in range(height)]

    # 이미지의 각 픽셀을 순회하며 R, G, B 값을 추출하여 저장
    for i in range(height):
        for j in range(width):
            R, G, B = image[i][j]
            R_channel[i][j] = R
            G_channel[i][j] = G
            B_channel[i][j] = B

    return R_channel, G_channel, B_channel




#OpenCV로 원본 이미지를 BGR(Blue, Green, Red) 형식으로 읽어오기
BGR_src_img = cv2.imread('C:/Users/test.bmp')


# BGR에서 YCbCr로 변환
YCbCr_image = convert_to_YCbCr(BGR_src_img)

# Y 채널 히스토그램 평활화
Y_channel = [[pixel[0] for pixel in row] for row in YCbCr_image]
equalized_Y = histogram_equalization(Y_channel)

# 히스토그램 평활화된 Y 채널과 원래의 Cr, Cb 채널을 다시 합침
equalized_YCbCr = [[[equalized_Y[i][j]] + [YCbCr_image[i][j][1], YCbCr_image[i][j][2]] for j in range(len(YCbCr_image[0]))] for i in range(len(YCbCr_image))]

# 결과 이미지를 RGB 색 공간으로 다시 변환
equalized_image = convert_to_RGB(equalized_YCbCr)

# 원본 이미지를 RGB 형식으로 변환
RGB_img_src = convert_to_RGB(YCbCr_image)



# 이미지 및 히스토그램, CDF를 나란히 표시
fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2행 3열의 서브플롯 생성

#matplotlib는 RGB로 이미지를 읽어오기 때문에 RGB_img_src로 원본 이미지 불러오기
axs[0, 0].imshow(RGB_img_src)
axs[0, 0].axis('off')
axs[0, 0].set_title('Original Image')

axs[1, 0].imshow(equalized_image)
axs[1, 0].axis('off')
axs[1, 0].set_title('Result Image')

# 원본 이미지 R, G, B 채널 분리
R, G, B = split_channels(RGB_img_src)

# 원본 이미지 R, G, B 채널에 대한 히스토그램 계산
histogram_R = calculate_histogram(R)
histogram_G = calculate_histogram(G)
histogram_B = calculate_histogram(B)

# 원본 이미지 R, G, B 채널에 대한 CDF 계산
cdf_R = calculate_cdf(histogram_R)
cdf_G = calculate_cdf(histogram_G)
cdf_B = calculate_cdf(histogram_B)

# R, G, B 히스토그램을 각각 빨간색, 초록색, 파란색으로 나타냄
width = 1.0  # 막대의 너비
r_bins = [x - width for x in range(256)]
g_bins = [x for x in range(256)]
b_bins = [x + width for x in range(256)]

# 원본 이미지 R, G, B 히스토그램 그리기
# axs는 서브플롯을 배열 형태로 관리하는 데 사용되는 변수
# bar 함수는 막대 그래프를 그리는 함수, alpha는 그래프의 투명도, label은 라벨에 대한 설명 -> legend 함수로 표시
axs[0, 1].bar(r_bins, histogram_R, width=width, color='r', alpha=0.6, label='Red')
axs[0, 1].bar(g_bins, histogram_G, width=width, color='g', alpha=0.6, label='Green')
axs[0, 1].bar(b_bins, histogram_B, width=width, color='b', alpha=0.6, label='Blue')

axs[0, 1].set_title('Original Histogram')
axs[0, 1].legend() #label을 히스토그램에 띄어주는 역할

# 결과이미지 R, G, B 채널 분리
R, G, B = split_channels(equalized_image)

# 결과 이미지 R, G, B 채널에 대한 히스토그램 계산
histogram_R = calculate_histogram(R)
histogram_G = calculate_histogram(G)
histogram_B = calculate_histogram(B)

# 결과 이미지의 R, G, B 채널에 대한 CDF 계산
cdf_R_result = calculate_cdf(histogram_R)
cdf_G_result = calculate_cdf(histogram_G)
cdf_B_result = calculate_cdf(histogram_B)

# R, G, B 히스토그램을 각각 빨간색, 초록색, 파란색으로 나타냄
width = 1.0  # 막대의 너비
r_bins = [x - width for x in range(256)]
g_bins = [x for x in range(256)]
b_bins = [x + width for x in range(256)]

# 결과 이미지 R, G, B 히스토그램 그리기
axs[1, 1].bar(r_bins, histogram_R, width=width, color='r', alpha=0.6, label='Red')
axs[1, 1].bar(g_bins, histogram_G, width=width, color='g', alpha=0.6, label='Green')
axs[1, 1].bar(b_bins, histogram_B, width=width, color='b', alpha=0.6, label='Blue')
axs[1, 1].set_title('Result Histogram')
axs[1, 1].legend()



# 원본 이미지의 R, G, B 채널에 대한 CDF 그리기
# plot는 그래프를 그리는 함수
axs[0, 2].set_title('Original Image CDF')
axs[0, 2].plot(cdf_R, color='red', label='Red CDF')
axs[0, 2].plot(cdf_G, color='green', label='Green CDF')
axs[0, 2].plot(cdf_B, color='blue', label='Blue CDF')
axs[0, 2].set_title('Original Image CDF')
axs[0, 2].legend()

# 결과 이미지의 R, G, B 채널에 대한 CDF 그리기
axs[1, 2].set_title('Result Image CDF')
axs[1, 2].plot(cdf_R_result, color='red', label='Red CDF')
axs[1, 2].plot(cdf_G_result, color='green', label='Green CDF')
axs[1, 2].plot(cdf_B_result, color='blue', label='Blue CDF')
axs[1, 2].set_title('Result Image CDF')
axs[1, 2].legend()


plt.show()

