import numpy as np
import sys
import os
import random
import time
import ctypes


# --- 关键修改 ---
# 将项目根目录（当前文件所在目录的上一级）添加到 Python 的模块搜索路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import compressed_segmentation

try:
    import compressed_segmentation
except ImportError:
    print("错误：无法导入 'compressed_segmentation' 模块。")
    print(f"请确保您已经在项目根目录 '{project_root}' 运行了编译命令：")
    print("python setup.py build_ext --inplace")
    sys.exit(1)


def create_sample_data(shape, order='F', dtype=np.uint64):
    """
    创建一个每个位置都有唯一值的数据体，并强制使用指定的内存布局。
    
    注意: shape 现在应该是 4D (X, Y, Z, C)。
    """
    total_elements = np.prod(shape)
    if total_elements > np.iinfo(dtype).max:
        # 为了安全，我们只使用 max_val 的一小部分来确保标签表不会溢出
        max_label = 2**20 
        if total_elements > max_label:
            print(f"警告: 数组大小 {total_elements} 远超 {max_label}，可能导致压缩溢出检查失败。")
    
    print(f"正在创建尺寸为 {shape}，类型为 {dtype.__name__}，order 为 {order} 的样本数据...")
    
    # 使用 np.arange 创建连续值，确保每个体素都有独特的标签进行验证
    data = np.arange(total_elements, dtype=dtype).reshape(shape, order=order)
    
    return data


def test_decompress(compressed_data, original_data, data_shape, data_type, order, block_size):
    print(f"\n--- [2] 正在调用 C++ 全量解压缩函数 (order={order}) ---")
    
    # decompress 期望 3D 形状 (X,Y,Z)，除非是多通道，这里我们将 4D 形状传入
    decompressed_data = compressed_segmentation.decompress(
        compressed_data,
        volume_size=data_shape, # 传入 4D 形状
        dtype=data_type,
        order=order,
        block_size=block_size
    )
    print("解压缩完成！")

    if np.array_equal(original_data, decompressed_data):
        print(f"✅ 成功：全量解压缩后的数据与原始数据完全一致！")
        return True
    else:
        print(f"❌ 失败：全量解压缩后的数据与原始数据不匹配！")
        diff = np.where(original_data != decompressed_data)
        if len(diff[0]) > 0:
            # 找到第一个不匹配的位置。注意：diff 可能是 4 维索引
            first_diff_idx = tuple(d[0] for d in diff)
            print(f"总共有 {len(diff[0])} 个不匹配的位置。")
            print(f"第一个不匹配的位置（全局坐标）：{first_diff_idx}")
            print(f" - 期望的值 (来自原始数据): {original_data[first_diff_idx]}")
            print(f" - 解压得到的值: {decompressed_data[first_diff_idx]}")
        return False


def get_random_test_cases(data_shape_3D, num_cases=10):
    """
    随机生成 request_start 和 request_end。
    
    注意: 只使用 X, Y, Z 维度来生成坐标。
    """
    test_cases = []
    x_max, y_max, z_max = data_shape_3D

    # 包含一些特殊情况的固定测试用例
    fixed_cases = [
        {"name": "完全在块内", "request_start": (47, 53, 96), "request_end": (49, 55, 97)},
        {"name": "跨越块边界", "request_start": (2, 7, 10), "request_end": (6, 10, 15)},
        {"name": "与块边界对齐", "request_start": (0, 0, 0), "request_end": (8, 8, 4)},
        # 修复：确保零大小和无重叠区域不会通过 'continue' 被跳过
        {"name": "零大小区域", "request_start": (10, 10, 10), "request_end": (10, 10, 10)},
        {"name": "完全不重叠", "request_start": (x_max + 1, y_max + 1, z_max + 1), "request_end": (x_max + 2, y_max + 2, z_max + 2)},
        {"name": "全量解压", "request_start": (0, 0, 0), "request_end": data_shape_3D},
    ]
    test_cases.extend(fixed_cases)

    for i in range(num_cases):
        # 随机区域测试
        x_start = random.randint(0, x_max)
        y_start = random.randint(0, y_max)
        z_start = random.randint(0, z_max)
        
        # 确保 end >= start 且不超过 max
        x_end = random.randint(x_start, x_max)
        y_end = random.randint(y_start, y_max)
        z_end = random.randint(z_start, z_max)

        test_cases.append({
            "name": f"随机区域 {i + 1}",
            "request_start": (x_start, y_start, z_start),
            "request_end": (x_end, y_end, z_end)
        })

    return test_cases


def run_partial_tests(compressed_data, original_data, data_shape, data_type, order, block_size):
    print("\n--- [3] 正在执行局部解压 (非 In-Place) 测试 ---")
    all_passed = True

    # 3D/4D 形状处理
    data_shape_3D = data_shape[:3]
    
    chunk_start = (0, 0, 0)
    chunk_end = data_shape_3D
    print(f"--- 正在测试单一完整 chunk: (start={chunk_start}, end={chunk_end}) ---")

    test_cases = get_random_test_cases(data_shape_3D, num_cases=20)
    
    for test in test_cases:
        request_start = test['request_start']
        request_end = test['request_end']
        # 确保 request_end > request_start，否则无法进行有效测试
        if all(request_end[i] <= request_start[i] for i in range(3)):
             continue # 跳过零大小的请求

        try:
            # 调用 decompress_partial 函数
            partial_decompressed_data, xx, yy = compressed_segmentation.decompress_partial(
                compressed_data,
                volume_size=data_shape, # 传入 4D 形状
                dtype=data_type,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                request_start=request_start,
                request_end=request_end,
                block_size=block_size,
                order=order,
            )
            
            # 从原始 4D 数据中切片出期望的结果
            expected_data_slice = original_data[
                request_start[0]:request_end[0],
                request_start[1]:request_end[1],
                request_start[2]:request_end[2],
                ... # 确保包含所有通道
            ]

            is_equal = np.array_equal(partial_decompressed_data, expected_data_slice)

            if not is_equal:
                print(f"❌ 失败: {test['name']}")
                # ... (错误打印部分保持不变)
                all_passed = False
            else:
                print(f"✅ 成功: {test['name']} (request_start={request_start}, request_end={request_end})")
        
        except Exception as e:
            print(f"❌ 异常: {test['name']} 错误: {e}")
            all_passed = False

    return all_passed


def run_partial_in_place_tests(compressed_data, original_data, data_shape, data_type, order, block_size):
    print("\n--- [4] 正在执行局部解压 (In-Place) 测试 ---")
    all_passed = True

    data_shape_3D = data_shape[:3]
    volume_size = data_shape 

    # 假设 chunk 实际范围
    chunk_start = (0, 0, 0)
    chunk_end = data_shape_3D 
    chunk_start = (26752, 8064, 3808)
    chunk_end   = (26880, 8192, 3840)

    test_cases = get_random_test_cases(data_shape_3D, num_cases=20)

    for test in test_cases:
        # 假设 request 区域
        request_start = test['request_start']
        request_end = test['request_end']
        request_start = (26676, 8024, 3811)
        request_end   = (26776, 8124, 3911)

        # -------------------------
        # 计算交集 (intersection)
        # -------------------------
        intersection_start = tuple(max(chunk_start[i], request_start[i]) for i in range(3))
        intersection_end   = tuple(min(chunk_end[i],   request_end[i])   for i in range(3))

        # 如果没有交集，直接跳过
        if any(intersection_start[i] >= intersection_end[i] for i in range(3)):
            print(f"⚠️ 跳过: {test['name']} (无交集)")
            continue

        # 计算请求区域尺寸
        requested_shape_3D = tuple(request_end[i] - request_start[i] for i in range(3))
        requested_shape = requested_shape_3D + (volume_size[3],)

        # 分配输出数组
        output_array = np.full(
            requested_shape, 
            random.randint(99999, 100000), 
            dtype=data_type, 
            order=order
        )
        data_address = output_array.ctypes.data

        # print(f"数组的 NumPy 对象地址: {id(output_array)}")
        # print(f"数组数据的内存首地址 (Python int): {data_address}")

        # # 你也可以将它格式化为十六进制，这更符合内存地址的常见表示方式
        # print(f"数组数据的内存首地址 (十六进制): {hex(data_address)}")
        try:
            # --- 调用 decompress_partial_in_place ---
            compressed_segmentation.decompress_partial_in_place(
                compressed_data,
                volume_size=volume_size, # 4D
                dtype=data_type,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                request_start=request_start,
                request_end=request_end,
                output_array=output_array, # 4D 数组
                block_size=block_size,
                order=order,
            )

            # -------------------------
            # 取 expected_output 的交集部分
            # -------------------------
            expected_output = original_data[
                intersection_start[0]-chunk_start[0]:intersection_end[0]-chunk_start[0],
                intersection_start[1]-chunk_start[1]:intersection_end[1]-chunk_start[1],
                intersection_start[2]-chunk_start[2]:intersection_end[2]-chunk_start[2],
                ...
            ]

            # 从 output_array 中取对应的交集区域
            # 偏移量 = intersection_start - request_start
            offset = tuple(intersection_start[i] - request_start[i] for i in range(3))
            offset_end = tuple(offset[i] + (intersection_end[i] - intersection_start[i]) for i in range(3))

            output_slice = output_array[
                offset[0]:offset_end[0],
                offset[1]:offset_end[1],
                offset[2]:offset_end[2],
                ...
            ]

            # 比较交集部分
            is_equal = np.array_equal(output_slice, expected_output)

            if is_equal:
                print(f"✅ 成功: {test['name']} (In-Place) (intersection={intersection_start}->{intersection_end})")
            else:
                print(f"❌ 失败: {test['name']} (In-Place)")
                print(f"chunk=({chunk_start}->{chunk_end}), request=({request_start}->{request_end}), "
                      f"intersection=({intersection_start}->{intersection_end})")
                all_passed = False

        except Exception as e:
            print(f"❌ 异常: {test['name']} (In-Place) 错误: {e}")
            all_passed = False

    return all_passed

### 新增的并行测试函数
def run_partial_in_place_parallel_tests(compressed_data, original_data, data_shape, data_type, order, block_size, num_threads=4):
    print(f"\n--- [5] 正在执行局部解压 (In-Place, 并行={num_threads}) 测试 ---")
    
    data_shape_3D = data_shape[:3]
    volume_size = data_shape 
    chunk_start = (0, 0, 0)
    chunk_end = data_shape_3D
    # chunk_start = (26752, 8064, 3808)
    # chunk_end   = (26880, 8192, 3840)
    # 生成多个请求
    test_cases = get_random_test_cases(data_shape_3D, num_cases=40)
    requests = []
    for test in test_cases:

        request_start = test['request_start']
        request_end = test['request_end']
        # request_start = (26676, 8024, 3811)
        # request_end   = (26776, 8124, 3911)
        requested_shape_3D = tuple(request_end[i] - request_start[i] for i in range(3))
        if any(size <= 0 for size in requested_shape_3D):
            continue
            
        requested_shape = requested_shape_3D + (volume_size[3],)
        
        output_array = np.full(
            requested_shape, 
            random.randint(99999, 100000), 
            dtype=data_type, 
            order=order
        )
        # data_address = output_array.ctypes.data

        # print(f"数组的 NumPy 对象地址: {id(output_array)}")
        # print(f"数组数据的内存首地址 (Python int): {data_address}")

        # # 你也可以将它格式化为十六进制，这更符合内存地址的常见表示方式
        # print(f"数组数据的内存首地址 (十六进制): {hex(data_address)}")
        request_dict = {
            'encoded': compressed_data,
            'volume_size': volume_size,
            'dtype': data_type,
            'chunk_start': chunk_start,
            'chunk_end': chunk_end,
            'request_start': request_start,
            'request_end': request_end,
            'output_array': output_array,
            'block_size': block_size,
            'output_array_ptr': output_array.ctypes.data,
            'output_array_ndim':output_array.ndim,
            'output_array_shape':output_array.shape,
            'output_array_strides':output_array.strides,
        }
        # print(f"volume_size={volume_size},chunk_start={chunk_start}, chunk_end={chunk_end}, request_start={request_start},request_end={request_end}")
        requests.append(request_dict)
    
    # 一次性调用并行解压函数
    compressed_segmentation.decompress_partial_in_place_parallel(
        requests,
        parallel=num_threads,
        order=order
    )
    print("并行解压调用完成，正在验证结果...")

    # 逐一验证每个请求的结果
    for i, req in enumerate(requests):
        request_start = req['request_start']
        request_end = req['request_end']
        output_array = req['output_array']
        test_name = test_cases[i]['name']

        intersection_start = tuple(max(chunk_start[j], request_start[j]) for j in range(3))
        intersection_end = tuple(min(chunk_end[j], request_end[j]) for j in range(3))
        
        # 如果交集为空，我们不应该有任何数据被写入，所以跳过这个验证
        if any(intersection_start[j] >= intersection_end[j] for j in range(3)):
            print(f"✅ 成功: {test_name} (parallel) (request={request_start}->{request_end}) - 无交集")
            continue

        expected_output = original_data[
            intersection_start[0]:intersection_end[0],
            intersection_start[1]:intersection_end[1],
            intersection_start[2]:intersection_end[2],
            ...
        ]

        offset = tuple(intersection_start[j] - request_start[j] for j in range(3))
        output_slice = output_array[
            offset[0]:offset[0] + (intersection_end[0] - intersection_start[0]),
            offset[1]:offset[1] + (intersection_end[1] - intersection_start[1]),
            offset[2]:offset[2] + (intersection_end[2] - intersection_start[2]),
            ...
        ]

        is_equal = np.array_equal(output_slice, expected_output)
        
        if not is_equal:
            # 抛出具体的错误，包含所有相关信息
            # 找出第一个不匹配的索引，用于更精确的调试
            diff_indices = np.where(output_slice != expected_output)
            if len(diff_indices[0]) > 0:
                first_diff_idx = tuple(d[0] for d in diff_indices)
                # 将相对索引转换为全局索引
                global_diff_idx = (
                    intersection_start[0] + first_diff_idx[0],
                    intersection_start[1] + first_diff_idx[1],
                    intersection_start[2] + first_diff_idx[2],
                )
            else:
                global_diff_idx = "N/A"

            error_message = (
                f"❌ 失败: {test_name} (parallel) - 数据不匹配\n"
                f"  - 请求区域: {request_start} -> {request_end}\n"
                f"  - 交集区域: {intersection_start} -> {intersection_end}\n"
                f"  - 第一个不匹配位置 (交集内部): {first_diff_idx}\n"
                f"  - 第一个不匹配位置 (全局坐标): {global_diff_idx}\n"
                f"  - 期望值: {expected_output[first_diff_idx]}\n"
                f"  - 实际值: {output_slice[first_diff_idx]}"
            )
            raise ValueError(error_message)
        
        print(f"✅ 成功: {test_name} (parallel) (request={request_start}->{request_end})")

    return True

def main():
    
    # 强制要求所有数据和接口形状为 4D (X, Y, Z, C)
    data_shape = (128, 128, 32, 1) 
    data_type = np.uint64
    block_size = (4, 8, 16) # 压缩块大小仍为 3D

    # --- 测试 F-order 数据 ---
    print("\n\n#############################################")
    print("##             开始 F-Order 测试           ##")
    print("#############################################")
    # original_data_f 使用 F-order 内存布局
    original_data_f = create_sample_data(shape=data_shape, order='F', dtype=data_type) 
    
    start_time = time.time()
    # 使用 order='F' 压缩
    compressed_data_f = compressed_segmentation.compress(original_data_f, order='F', block_size=block_size)
    print(f"压缩 F-order 数据耗时：{time.time() - start_time:.4f} 秒")

    if test_decompress(compressed_data_f, original_data_f, data_shape, data_type, order='F', block_size=block_size):
        # run_partial_tests(compressed_data_f, original_data_f, data_shape, data_type, order='F', block_size=block_size)
        run_partial_in_place_tests(compressed_data_f, original_data_f, data_shape, data_type, order='F', block_size=block_size)
        run_partial_in_place_parallel_tests(compressed_data_f, original_data_f, data_shape, data_type, order='F', block_size=block_size, num_threads=8)
    
    # --- 测试 C-order 数据 ---
    print("\n\n#############################################")
    print("##             开始 C-Order 测试           ##")
    print("#############################################")
    # original_data_c 使用 C-order 内存布局
    original_data_c = create_sample_data(shape=data_shape, order='C', dtype=data_type) 
    
    start_time = time.time()
    # 使用 order='C' 压缩
    compressed_data_c = compressed_segmentation.compress(original_data_c, order='C', block_size=block_size)
    print(f"压缩 C-order 数据耗时：{time.time() - start_time:.4f} 秒")

    if test_decompress(compressed_data_c, original_data_c, data_shape, data_type, order='C', block_size=block_size):
        # run_partial_tests(compressed_data_c, original_data_c, data_shape, data_type, order='C', block_size=block_size)
        # run_partial_in_place_tests(compressed_data_c, original_data_c, data_shape, data_type, order='C', block_size=block_size)
        run_partial_in_place_parallel_tests(compressed_data_c, original_data_c, data_shape, data_type, order='C', block_size=block_size, num_threads=8)


if __name__ == "__main__":
    main()