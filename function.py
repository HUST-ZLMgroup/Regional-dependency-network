import numpy as np
from scipy.stats import chi2


def calculate_ellipse(xy, weight):
    # 计算均值
    mean = np.average(xy, axis=0, weights=weight)
    # 计算加权协方差矩阵
    cov_matrix = np.cov(xy, rowvar=False, aweights=weight)
    # 协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # 计算椭圆长短半轴
    alpha = np.sqrt(eigenvalues[0])
    beta = np.sqrt(eigenvalues[1])
    # 计算椭圆角度
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    return mean[0], mean[1], alpha, beta, angle


def cal_curve(p1, p2, r=5):
    """
    计算连接两点之间的一条曲线
    :param p1: 起点，包含p1.x、p1.y属性
    :param p2: 终点，包含p2.x、p2.y属性
    :param r: 控制线形的参数
    :return: 曲线坐标
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    x0 = np.linspace(0, 1, 51)
    y0 = (- x0 ** 2 + x0) / r

    xx = x0 * dx - y0 * dy
    yy = x0 * dy + y0 * dx

    return xx + p1.x, yy + p1.y


def cal_dist(coords):
    dLon = np.radians(coords[:, [0]] - coords[:, [0]].T)
    dLat = np.radians(coords[:, [1]] - coords[:, [1]].T)
    lat = np.radians(coords[:, [1]])
    a = np.sin(dLat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dLon / 2) ** 2
    # a += np.eye(coords.shape[0])
    R = 6371.0

    dist = 2 * R * np.arcsin(np.sqrt(a))
    return dist


def cal_weight(coords, gdp=None, k: int = None):
    """
    用引力模型计算权重矩阵
    :param coords: N×2,经纬度坐标
    :param gdp: N,GDP
    :param k: k近邻
    :return: N×N,权重矩阵w
    """
    # w = []
    # for i in range(coords.shape[0]):
    #     dLon = np.radians(coords[:, 0] - coords[i, 0])
    #     dLat = np.radians(coords[:, 1] - coords[i, 1])
    #     lat1 = np.radians(coords[:, 1])
    #     lat2 = np.radians(coords[i, 1])
    #     a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
    #     R = 6371.0
    #
    #     a[i] = 1
    #     dist = 2 * R * np.arcsin(np.sqrt(a))
    #     if gdp is not None:
    #         g = gdp * gdp[i] / dist ** 2
    #     else:
    #         g = 1 / dist ** 2
    #     g[i] = 0
    #     w.append(g)

    # return np.vstack(w)

    dist = cal_dist(coords)

    if gdp is not None:
        gdp = gdp.reshape((-1, 1))
        w = gdp * gdp.T / dist ** 2
    else:
        w = 1 / dist ** 2
    for i in range(coords.shape[0]):
        w[i, i] = 0
    if k is not None:
        mask = dist > np.partition(dist, k - 1, axis=1)[:, [k - 1]]
        # w[~mask] = 1
        # w[~mask.T] = 1
        w[mask] = 0
        w[mask.T] = 0
    return w


def sigmoid(x, alpha=0.99, beta=0.95):
    a = x.flatten()
    arr_partitioned = np.partition(a, int(a.shape[0] * alpha))
    c = np.log(beta / (1 - beta)) / (arr_partitioned[int(a.shape[0] * alpha)] - np.median(a))
    b = 1 / (1 + np.exp((-x + np.median(a)) * c))
    return b


if __name__ == '__main__':
    print(chi2.ppf(0.6827, df=1))
    '''
    data = pd.read_csv('GDP.csv', encoding='GBK')
    coords = data[['lon', 'lat']].values[-274:]
    gdp = data['GDP(亿元)'].values[-274 * 5:].reshape((5, 274))
    gdp_mean = np.mean(gdp, axis=0)
    weights = cal_weight(coords, gdp_mean)
    weights = np.sqrt(weights / np.max(weights))
    print(np.where(weights == 1.))
    columns = [f'w{i + 1}' for i in range(274)]
    df = pd.DataFrame(weights, columns=columns)
    df.to_csv('w_sqrt.csv', index=False)
    
    data = pd.read_csv('data-5.csv', encoding='GBK')
    columns = data.columns
    ar = data.values
    ar[:, 4:] = np.log(ar[:, 4:])
    df = pd.DataFrame(ar, columns=columns)
    df.to_csv('data-5-ln.csv', index=False, encoding='GBK')
    '''
