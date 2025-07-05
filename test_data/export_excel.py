import pandas as pd
import numpy as np
import random
import openpyxl
from datetime import datetime, timedelta
from faker import Faker
import argparse

# 初始化伪造数据生成器
fake = Faker('zh_CN')  # 使用中文数据



def generate_student_data(rows):
    """生成学生成绩数据"""
    data = {
        '学号': [f'S{str(i).zfill(5)}' for i in range(1, rows + 1)],
        '姓名': [fake.name() for _ in range(rows)],
        '性别': [random.choice(['男', '女']) for _ in range(rows)],
        '班级': [f'{random.randint(1, 12)}年级{random.randint(1, 10)}班' for _ in range(rows)],
        '语文': [random.randint(40, 100) for _ in range(rows)],
        '数学': [random.randint(40, 100) for _ in range(rows)],
        '英语': [random.randint(40, 100) for _ in range(rows)],
        '总分': [0] * rows,
        '是否及格': [False] * rows
    }

    # 计算总分和是否及格
    for i in range(rows):
        total = data['语文'][i] + data['数学'][i] + data['英语'][i]
        data['总分'][i] = total
        data['是否及格'][i] = '是' if total >= 180 else '否'

    return pd.DataFrame(data)


def generate_sales_data(rows):
    """生成销售数据"""
    products = ['手机', '笔记本', '平板', '耳机', '充电器', '智能手表', '相机', '打印机']
    data = {
        '订单号': [f'ORD{str(i).zfill(6)}' for i in range(1, rows + 1)],
        '产品': [random.choice(products) for _ in range(rows)],
        '数量': [random.randint(1, 10) for _ in range(rows)],
        '单价': [round(random.uniform(100, 2000), 2) for _ in range(rows)],
        '销售日期': [fake.date_between(start_date='-1y', end_date='today') for _ in range(rows)],
        '销售员': [fake.name() for _ in range(rows)],
        '地区': [fake.city() for _ in range(rows)]
    }

    # 计算总金额
    data['总金额'] = [round(data['数量'][i] * data['单价'][i], 2) for i in range(rows)]

    return pd.DataFrame(data)


def generate_employee_data(rows):
    """生成员工数据"""
    departments = ['技术部', '销售部', '财务部', '人力资源', '市场部', '客服部']
    positions = ['经理', '主管', '工程师', '专员', '助理', '分析师']

    data = {
        '员工ID': [f'E{str(i).zfill(4)}' for i in range(1, rows + 1)],
        '姓名': [fake.name() for _ in range(rows)],
        '部门': [random.choice(departments) for _ in range(rows)],
        '职位': [random.choice(positions) for _ in range(rows)],
        '入职日期': [fake.date_between(start_date='-5y', end_date='-30d') for _ in range(rows)],
        '基本工资': [random.randint(6000, 15000) for _ in range(rows)],
        '绩效系数': [round(random.uniform(0.8, 1.5), 2) for _ in range(rows)],
        '邮箱': [fake.email() for _ in range(rows)]
    }

    # 计算总工资
    data['总工资'] = [round(data['基本工资'][i] * data['绩效系数'][i]) for i in range(rows)]

    return pd.DataFrame(data)




if __name__ == "__main__":
    # 直接调用生成函数
    df = generate_employee_data(500)
    df.to_excel('generate_employee_data.xlsx', index=False)
    print(f"成功生成学生数据到 generate_employee_data.xlsx")