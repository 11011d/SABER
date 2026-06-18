import numpy as np
import sys
import os
import random
import time
import ctypes


# --- Key change ---
# Add the project root (the parent directory of this file) to the Python module search path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import compressed_segmentation

try:
    import compressed_segmentation
except ImportError:
    print("Error: unable to import the 'compressed_segmentation' module.")
    print(f"Please make sure you have run the build command in the project root '{project_root}':")
    print("python setup.py build_ext --inplace")
    sys.exit(1)


def create_sample_data(shape, order='F', dtype=np.uint64):
    """
    Create a volume whose every position has a unique value and force the requested memory layout.
    
    Note: shape should now be 4D (X, Y, Z, C).
    """
    total_elements = np.prod(shape)
    if total_elements > np.iinfo(dtype).max:
        # For safety, use only a small portion of max_val to make sure the label table does not overflow.
        max_label = 2**20 
        if total_elements > max_label:
            print(f"Warning: array size {total_elements} greatly exceeds {max_label}, which may cause the compression overflow check to fail.")
    
    print(f"Creating sample data with shape {shape}, dtype {dtype.__name__}, and order {order}...")
    
    # Use np.arange to create consecutive values so every voxel has a unique label for validation.
    data = np.arange(total_elements, dtype=dtype).reshape(shape, order=order)
    
    return data


def test_decompress(compressed_data, original_data, data_shape, data_type, order, block_size):
    print(f"\n--- [2] Calling the full C++ decompression function (order={order}) ---")
    
    # decompress expects a 3D shape (X, Y, Z); for multi-channel data we pass the 4D shape here.
    decompressed_data = compressed_segmentation.decompress(
        compressed_data,
        volume_size=data_shape, # Pass the 4D shape.
        dtype=data_type,
        order=order,
        block_size=block_size
    )
    print("Decompression finished!")

    if np.array_equal(original_data, decompressed_data):
        print(f"SUCCESS: the fully decompressed data exactly matches the original data.")
        return True
    else:
        print(f"FAIL: the fully decompressed data does not match the original data.")
        diff = np.where(original_data != decompressed_data)
        if len(diff[0]) > 0:
            # Find the first mismatching position. Note that diff may be a 4D index.
            first_diff_idx = tuple(d[0] for d in diff)
            print(f"There are {len(diff[0])} mismatching positions in total.")
            print(f"First mismatch position (global coordinates): {first_diff_idx}")
            print(f" - Expected value (from the original data): {original_data[first_diff_idx]}")
            print(f" - Decompressed value: {decompressed_data[first_diff_idx]}")
        return False


def get_random_test_cases(data_shape_3D, num_cases=10):
    """
    Randomly generate request_start and request_end.
    
    Note: only use the X, Y, and Z dimensions to generate coordinates.
    """
    test_cases = []
    x_max, y_max, z_max = data_shape_3D

    # Include fixed test cases that cover special situations.
    fixed_cases = [
        {"name": "fully inside a block", "request_start": (47, 53, 96), "request_end": (49, 55, 97)},
        {"name": "crossing block boundaries", "request_start": (2, 7, 10), "request_end": (6, 10, 15)},
        {"name": "aligned with block boundaries", "request_start": (0, 0, 0), "request_end": (8, 8, 4)},
        # Ensure zero-size and non-overlapping regions are not skipped by 'continue'.
        {"name": "zero-size region", "request_start": (10, 10, 10), "request_end": (10, 10, 10)},
        {"name": "no overlap at all", "request_start": (x_max + 1, y_max + 1, z_max + 1), "request_end": (x_max + 2, y_max + 2, z_max + 2)},
        {"name": "full decompression", "request_start": (0, 0, 0), "request_end": data_shape_3D},
    ]
    test_cases.extend(fixed_cases)

    for i in range(num_cases):
        # Random region test.
        x_start = random.randint(0, x_max)
        y_start = random.randint(0, y_max)
        z_start = random.randint(0, z_max)
        
        # Ensure end >= start and does not exceed the maximum.
        x_end = random.randint(x_start, x_max)
        y_end = random.randint(y_start, y_max)
        z_end = random.randint(z_start, z_max)

        test_cases.append({
            "name": f"random region {i + 1}",
            "request_start": (x_start, y_start, z_start),
            "request_end": (x_end, y_end, z_end)
        })

    return test_cases


def run_partial_tests(compressed_data, original_data, data_shape, data_type, order, block_size):
    print("\n--- [3] Running partial decompression tests (non in-place) ---")
    all_passed = True

    # Handle 3D and 4D shapes.
    data_shape_3D = data_shape[:3]
    
    chunk_start = (0, 0, 0)
    chunk_end = data_shape_3D
    print(f"--- Testing a single full chunk: (start={chunk_start}, end={chunk_end}) ---")

    test_cases = get_random_test_cases(data_shape_3D, num_cases=20)
    
    for test in test_cases:
        request_start = test['request_start']
        request_end = test['request_end']
        # Ensure request_end > request_start, otherwise the test is not meaningful.
        if all(request_end[i] <= request_start[i] for i in range(3)):
             continue # Skip zero-size requests.

        try:
            # Call decompress_partial.
            partial_decompressed_data, xx, yy = compressed_segmentation.decompress_partial(
                compressed_data,
                volume_size=data_shape, # Pass the 4D shape.
                dtype=data_type,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                request_start=request_start,
                request_end=request_end,
                block_size=block_size,
                order=order,
            )
            
            # Slice out the expected result from the original 4D data.
            expected_data_slice = original_data[
                request_start[0]:request_end[0],
                request_start[1]:request_end[1],
                request_start[2]:request_end[2],
                ... # Make sure all channels are included.
            ]

            is_equal = np.array_equal(partial_decompressed_data, expected_data_slice)

            if not is_equal:
                print(f"FAIL: {test['name']}")
                # ... (keep the error-printing section unchanged)
                all_passed = False
            else:
                print(f"SUCCESS: {test['name']} (request_start={request_start}, request_end={request_end})")
        
        except Exception as e:
            print(f"ERROR: {test['name']} error: {e}")
            all_passed = False

    return all_passed


def run_partial_in_place_tests(compressed_data, original_data, data_shape, data_type, order, block_size):
    print("\n--- [4] Running partial decompression tests (in-place) ---")
    all_passed = True

    data_shape_3D = data_shape[:3]
    volume_size = data_shape 

    # Example chunk range.
    chunk_start = (0, 0, 0)
    chunk_end = data_shape_3D 
    chunk_start = (26752, 8064, 3808)
    chunk_end   = (26880, 8192, 3840)

    test_cases = get_random_test_cases(data_shape_3D, num_cases=20)

    for test in test_cases:
        # Example request range.
        request_start = test['request_start']
        request_end = test['request_end']
        request_start = (26676, 8024, 3811)
        request_end   = (26776, 8124, 3911)

        # -------------------------
        # Compute the intersection.
        # -------------------------
        intersection_start = tuple(max(chunk_start[i], request_start[i]) for i in range(3))
        intersection_end   = tuple(min(chunk_end[i],   request_end[i])   for i in range(3))

        # Skip directly if there is no intersection.
        if any(intersection_start[i] >= intersection_end[i] for i in range(3)):
            print(f"SKIP: {test['name']} (no intersection)")
            continue

        # Compute the requested region size.
        requested_shape_3D = tuple(request_end[i] - request_start[i] for i in range(3))
        requested_shape = requested_shape_3D + (volume_size[3],)

        # Allocate the output array.
        output_array = np.full(
            requested_shape, 
            random.randint(99999, 100000), 
            dtype=data_type, 
            order=order
        )
        data_address = output_array.ctypes.data

        # print(f"NumPy object address: {id(output_array)}")
        # print(f"Base memory address of the array data (Python int): {data_address}")

        # # You can also format it as hexadecimal, which is a more common way to display memory addresses.
        # print(f"Base memory address of the array data (hex): {hex(data_address)}")
        try:
            # --- Call decompress_partial_in_place ---
            compressed_segmentation.decompress_partial_in_place(
                compressed_data,
                volume_size=volume_size, # 4D
                dtype=data_type,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                request_start=request_start,
                request_end=request_end,
                output_array=output_array, # 4D array
                block_size=block_size,
                order=order,
            )

            # -------------------------
            # Extract the intersection part of expected_output.
            # -------------------------
            expected_output = original_data[
                intersection_start[0]-chunk_start[0]:intersection_end[0]-chunk_start[0],
                intersection_start[1]-chunk_start[1]:intersection_end[1]-chunk_start[1],
                intersection_start[2]-chunk_start[2]:intersection_end[2]-chunk_start[2],
                ...
            ]

            # Extract the corresponding intersection region from output_array.
            # Offset = intersection_start - request_start
            offset = tuple(intersection_start[i] - request_start[i] for i in range(3))
            offset_end = tuple(offset[i] + (intersection_end[i] - intersection_start[i]) for i in range(3))

            output_slice = output_array[
                offset[0]:offset_end[0],
                offset[1]:offset_end[1],
                offset[2]:offset_end[2],
                ...
            ]

            # Compare the intersection region.
            is_equal = np.array_equal(output_slice, expected_output)

            if is_equal:
                print(f"SUCCESS: {test['name']} (In-Place) (intersection={intersection_start}->{intersection_end})")
            else:
                print(f"FAIL: {test['name']} (In-Place)")
                print(f"chunk=({chunk_start}->{chunk_end}), request=({request_start}->{request_end}), "
                      f"intersection=({intersection_start}->{intersection_end})")
                all_passed = False

        except Exception as e:
            print(f"ERROR: {test['name']} (In-Place) error: {e}")
            all_passed = False

    return all_passed

### Additional parallel test function
def run_partial_in_place_parallel_tests(compressed_data, original_data, data_shape, data_type, order, block_size, num_threads=4):
    print(f"\n--- [5] Running partial decompression tests (in-place, parallel={num_threads}) ---")
    
    data_shape_3D = data_shape[:3]
    volume_size = data_shape 
    chunk_start = (0, 0, 0)
    chunk_end = data_shape_3D
    # chunk_start = (26752, 8064, 3808)
    # chunk_end   = (26880, 8192, 3840)
    # Generate multiple requests.
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

        # print(f"NumPy object address: {id(output_array)}")
        # print(f"Base memory address of the array data (Python int): {data_address}")

        # # You can also format it as hexadecimal, which is a more common way to display memory addresses.
        # print(f"Base memory address of the array data (hex): {hex(data_address)}")
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
    
    # Call the parallel decompression function in one shot.
    compressed_segmentation.decompress_partial_in_place_parallel(
        requests,
        parallel=num_threads,
        order=order
    )
    print("Parallel decompression call finished, validating results...")

    # Validate each request result one by one.
    for i, req in enumerate(requests):
        request_start = req['request_start']
        request_end = req['request_end']
        output_array = req['output_array']
        test_name = test_cases[i]['name']

        intersection_start = tuple(max(chunk_start[j], request_start[j]) for j in range(3))
        intersection_end = tuple(min(chunk_end[j], request_end[j]) for j in range(3))
        
        # If the intersection is empty, nothing should have been written, so skip this validation.
        if any(intersection_start[j] >= intersection_end[j] for j in range(3)):
            print(f"SUCCESS: {test_name} (parallel) (request={request_start}->{request_end}) - no intersection")
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
            # Raise a detailed error containing all relevant information.
            # Find the first mismatching index for more precise debugging.
            diff_indices = np.where(output_slice != expected_output)
            if len(diff_indices[0]) > 0:
                first_diff_idx = tuple(d[0] for d in diff_indices)
                # Convert the relative index to a global index.
                global_diff_idx = (
                    intersection_start[0] + first_diff_idx[0],
                    intersection_start[1] + first_diff_idx[1],
                    intersection_start[2] + first_diff_idx[2],
                )
            else:
                global_diff_idx = "N/A"

            error_message = (
                f"FAIL: {test_name} (parallel) - data mismatch\n"
                f"  - request region: {request_start} -> {request_end}\n"
                f"  - intersection region: {intersection_start} -> {intersection_end}\n"
                f"  - first mismatching position (inside the intersection): {first_diff_idx}\n"
                f"  - first mismatching position (global coordinates): {global_diff_idx}\n"
                f"  - expected value: {expected_output[first_diff_idx]}\n"
                f"  - actual value: {output_slice[first_diff_idx]}"
            )
            raise ValueError(error_message)
        
        print(f"SUCCESS: {test_name} (parallel) (request={request_start}->{request_end})")

    return True

def main():
    
    # Force all data and interface shapes to be 4D (X, Y, Z, C).
    data_shape = (128, 128, 32, 1) 
    data_type = np.uint64
    block_size = (4, 8, 16) # The compression block size remains 3D.

    # --- Test F-order data ---
    print("\n\n#############################################")
    print("##             Starting F-Order Test        ##")
    print("#############################################")
    # original_data_f uses an F-order memory layout.
    original_data_f = create_sample_data(shape=data_shape, order='F', dtype=data_type) 
    
    start_time = time.time()
    # Compress with order='F'.
    compressed_data_f = compressed_segmentation.compress(original_data_f, order='F', block_size=block_size)
    print(f"Time spent compressing F-order data: {time.time() - start_time:.4f} seconds")

    if test_decompress(compressed_data_f, original_data_f, data_shape, data_type, order='F', block_size=block_size):
        # run_partial_tests(compressed_data_f, original_data_f, data_shape, data_type, order='F', block_size=block_size)
        run_partial_in_place_tests(compressed_data_f, original_data_f, data_shape, data_type, order='F', block_size=block_size)
        run_partial_in_place_parallel_tests(compressed_data_f, original_data_f, data_shape, data_type, order='F', block_size=block_size, num_threads=8)
    
    # --- Test C-order data ---
    print("\n\n#############################################")
    print("##             Starting C-Order Test        ##")
    print("#############################################")
    # original_data_c uses a C-order memory layout.
    original_data_c = create_sample_data(shape=data_shape, order='C', dtype=data_type) 
    
    start_time = time.time()
    # Compress with order='C'.
    compressed_data_c = compressed_segmentation.compress(original_data_c, order='C', block_size=block_size)
    print(f"Time spent compressing C-order data: {time.time() - start_time:.4f} seconds")

    if test_decompress(compressed_data_c, original_data_c, data_shape, data_type, order='C', block_size=block_size):
        # run_partial_tests(compressed_data_c, original_data_c, data_shape, data_type, order='C', block_size=block_size)
        # run_partial_in_place_tests(compressed_data_c, original_data_c, data_shape, data_type, order='C', block_size=block_size)
        run_partial_in_place_parallel_tests(compressed_data_c, original_data_c, data_shape, data_type, order='C', block_size=block_size, num_threads=8)


if __name__ == "__main__":
    main()
