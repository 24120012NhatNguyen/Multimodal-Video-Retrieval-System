import numpy as np
from typing import List, Tuple

def dp_alignment(pts_times: List[float], event_scores: np.ndarray, delta: float = 5.0, gamma: float = 0.5) -> Tuple[List[int], float]:
    """
    Tìm chuỗi frame khớp nhất với chuỗi sự kiện bằng Quy hoạch động (DP).
    Cho phép "skip" 1 sự kiện nếu bị hụt, chịu phạt điểm gamma.
    
    Args:
        pts_times: Danh sách thời gian (giây) của các frame trong video.
        event_scores: Ma trận điểm (num_frames, num_events) cho từng sự kiện.
        delta: Khoảng cách thời gian tối đa giữa 2 sự kiện liên tiếp (giây).
        gamma: Điểm phạt khi skip 1 sự kiện.
        
    Returns:
        best_path: Danh sách các chỉ số frame trong pts_times đại diện cho chuỗi sự kiện tìm được.
        max_score: Tổng điểm của chuỗi.
    """
    pts_times = np.array(pts_times)
    event_scores = np.array(event_scores)
    
    num_frames, num_events = event_scores.shape
    if num_events == 0 or num_frames == 0:
        return [], -np.inf
        
    dp = np.full((num_frames, num_events), -np.inf)
    trace = np.full((num_frames, num_events), -1, dtype=int)
    skip_trace = np.full((num_frames, num_events), False, dtype=bool)
    
    # Khởi tạo cho sự kiện đầu tiên (k=0)
    dp[:, 0] = event_scores[:, 0]
    
    for k in range(1, num_events):
        for i in range(num_frames):
            t_i = pts_times[i]
            
            max_val = -np.inf
            best_j = -1
            skipped = False
            
            # Tìm frame j trước đó thỏa mãn khoảng cách thời gian (không skip)
            for j in range(i):
                t_j = pts_times[j]
                if t_i - delta <= t_j < t_i:
                    if dp[j, k-1] > max_val:
                        max_val = dp[j, k-1]
                        best_j = j
            
            # Chấp nhận skip sự kiện k-1 (bị phạt gamma)
            if k >= 2:
                for j in range(i):
                    t_j = pts_times[j]
                    if t_i - 2 * delta <= t_j < t_i:
                        val = dp[j, k-2] - gamma
                        if val > max_val:
                            max_val = val
                            best_j = j
                            skipped = True
                            
            if max_val != -np.inf:
                dp[i, k] = event_scores[i, k] + max_val
                trace[i, k] = best_j
                skip_trace[i, k] = skipped
                
    # Tìm frame kết thúc có điểm cao nhất cho sự kiện cuối cùng
    best_end = int(np.argmax(dp[:, -1]))
    max_score = float(dp[best_end, -1])
    
    if max_score == -np.inf:
        return [], -np.inf
        
    # Backtrack tìm chuỗi frame tối ưu nhất
    path = []
    curr = best_end
    curr_k = num_events - 1
    
    while curr_k >= 0 and curr != -1:
        path.append(curr)
        skipped = skip_trace[curr, curr_k]
        curr = trace[curr, curr_k]
        if skipped:
            # Nếu skip sự kiện k-1, ta thêm giá trị -1 vào path để biểu diễn frame bị thiếu
            path.append(-1)
            curr_k -= 2
        else:
            curr_k -= 1
            
    # Lật ngược đường đi vì ta đi từ cuối lên đầu
    path.reverse()
    
    return path, max_score
