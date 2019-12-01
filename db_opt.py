'''
@description：# 数据库查询程序，数据已经通过处理并放在了服务器中的mysql数据库中，四个字段分别表示的是id，颜色，车型和车辆id
@time：2019年11月30日20:01:39
@author：西安交通大学软件学院场景检索与检测组
'''
import pymysql


# 测试是否可以连接服务器
def search_version():
    db = pymysql.connect("139.199.193.78", "root", "123456", "vehicle")

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    cursor.execute("SELECT VERSION()")

    # 使用 fetchone() 方法获取单条数据.
    data = cursor.fetchone()

    print("Database version : %s " % data)

    # 关闭数据库连接
    db.close()


# 通过预测ID查询图片路径，返回路径
def get_paths_by_id(vid):
    db = pymysql.connect("139.199.193.78", "root", "123456", "vehicle")

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    sql = "select pid from vehicle.vehicle_info where vehicle_id = %d" % (vid)
    paths = []
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        for resultx in results:
            # print(resultx)
            result = resultx[0]
            result = str(result)
            length = len(result)
            # print(length)
            for i in range(7-length):
                result = '0' + result
            result = result + '.jpg'
            paths.append(result)
    except:
        print("Error: unable to fetch data")

    return paths


if __name__ == '__main__':
    results = get_paths_by_id(5049)
    print(results)
    # print(len(str(7888)))
