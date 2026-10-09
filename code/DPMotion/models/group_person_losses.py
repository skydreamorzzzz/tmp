"""
Group/Person 交互的距离损失函数

这个模块包含所有Group/Person交互相关的距离约束损失
"""

import torch as th


def parse_distance_constraint(constraint_str):
    """
    解析距离约束字符串
    
    Args:
        constraint_str: 如 ">=1.2m and <=3m" 或 ">=3.5m"
        
    Returns:
        (lower_bound, upper_bound): 距离范围（米）
            如果只有下界，upper_bound=inf
    """
    import re
    if constraint_str is None:
        return None, None
    
    constraint_str = constraint_str.replace(" ", "").lower()
    numbers = re.findall(r'\d+\.?\d*', constraint_str)
    numbers = [float(n) for n in numbers]
    
    if not numbers:
        return None, None
    
    if "and" in constraint_str:
        # 范围约束: ">=1.2m and <=3m"
        lower = numbers[0] if ">=" in constraint_str.split("and")[0] else 0.0
        upper = numbers[1] if len(numbers) > 1 else float('inf')
        return lower, upper
    else:
        # 单边约束
        if "<=" in constraint_str:
            return 0.0, numbers[0]
        elif ">=" in constraint_str:
            return numbers[0], float('inf')
    
    return None, None


def soft_range_loss(distances, lower, upper):
    """
    软区间损失：距离在[lower, upper]范围内损失为0，超出范围平方惩罚
    
    Args:
        distances: (B, T) 距离张量
        lower: 下界
        upper: 上界
        
    Returns:
        loss: 标量损失值
    """
    if lower is None:
        return th.tensor(0.0, device=distances.device, dtype=distances.dtype)
    
    # 处理只有下界的情况（upper=inf）
    if upper == float('inf'):
        # 只惩罚小于lower的情况
        violation = th.clamp(lower - distances, min=0.0)
        return (th.exp(violation) - 1.0).mean()
    
    # 完整区间约束
    mid = (lower + upper) / 2.0
    margin = (upper - lower) / 2.0
    diff = distances - mid
    excess = (diff.abs() - margin).clamp(min=0.0)
    return (th.exp(excess) - 1.0).mean()


def extract_root_positions(x_start_list):
    """
    从x_start列表中提取所有人的root位置
    
    Args:
        x_start_list: [x_start_0, x_start_1, ...] 每个x_start shape为 (B, T, D)
        
    Returns:
        root_positions: list of (B, T, 3)，每个人的root位置
    """
    roots = []
    for x_start in x_start_list:
        B, T = x_start.shape[:2]
        joints = x_start[..., :22 * 3].reshape(B, T, 22, 3)
        root = joints[..., 0, :]  # (B, T, 3)
        roots.append(root)
    return roots


def compute_distance_xz(root1, root2):
    """
    计算XZ平面上的距离
    
    Args:
        root1: (B, T, 3)
        root2: (B, T, 3)
        
    Returns:
        distances: (B, T)
    """
    rel_pos = root1 - root2  # (B, T, 3)
    rel_pos_xz = rel_pos[..., [0, 2]]  # (B, T, 2)
    distances = th.norm(rel_pos_xz, dim=-1)  # (B, T)
    return distances


def extract_joints_list(x_start_list):
    """
    从x_start列表中提取所有人的关节位置
    
    Args:
        x_start_list: [x_start_0, x_start_1, ...] 每个x_start shape为 (B, T, D)
        
    Returns:
        joints_list: list of (B, T, 22, 3)
    """
    joints_list = []
    for x_start in x_start_list:
        B, T = x_start.shape[:2]
        joints = x_start[..., :22 * 3].reshape(B, T, 22, 3)
        joints_list.append(joints)
    return joints_list


def compute_forward_direction(joints):
    """
    计算此时刻的身体朝向向量 (在XZ平面上)
    使用 Root(0), Left Collar(13), Right Collar(14) 计算身体正面朝向，与可视化逻辑保持一致
    
    Args:
        joints: (B, T, 22, 3)
    """
    # joint_0: Root
    joint_0 = joints[..., 0, :]
    # joint_13: Left Collar
    joint_13 = joints[..., 13, :]
    # joint_14: Right Collar
    joint_14 = joints[..., 14, :]
    
    # 向量1: Root -> Left Collar (大致向上偏左)
    vec1 = joint_13 - joint_0
    # 向量2: Left Collar -> Right Collar (大致水平向右)
    vec2 = joint_14 - joint_13
    
    # 叉乘得到法向量 (身体正面朝向)
    forward = th.cross(vec1, vec2, dim=-1)
    
    # 投影到XZ平面并归一化
    forward_xz = forward[..., [0, 2]]  # (B, T, 2) 只取XZ平面 [x, z]
    forward_xz_norm = th.norm(forward_xz, dim=-1, keepdim=True) + 1e-8
    forward_xz = forward_xz / forward_xz_norm  # 归一化
    
    return forward_xz


def compute_direction_vector_xz(point1, point2):
    """
    计算从point1到point2的单位向量 (在XZ平面上)
    
    Args:
        point1: (B, T, 3) 起点
        point2: (B, T, 3) 终点
        
    Returns:
        direction: (B, T, 2) 单位向量
    """
    diff = point2 - point1
    diff_xz = diff[..., [0, 2]]
    direction = diff_xz / (th.norm(diff_xz, dim=-1, keepdim=True) + 1e-6)
    return direction


def compute_group_person_losses(all_starts, inter_graph, weight=5.0):
    """
    计算所有Group/Person交互的距离损失
    
    Args:
        all_starts: list of x_start，每个shape为(B, T, D)
        inter_graph: 交互图，包含groups, person_to_group, distance_constraints等
        weight: 损失权重
        
    Returns:
        total_loss: 总损失
        loss_dict: 各个损失分项的字典（用于日志）
    """
    device = all_starts[0].device
    total_loss = th.tensor(0.0, device=device, dtype=all_starts[0].dtype)
    loss_dict = {}
    
    # 提取所有人的root位置
    roots = extract_root_positions(all_starts)
    # 提取所有人的关节位置 (用于计算朝向)
    joints_list = extract_joints_list(all_starts)
    
    # 1. Group内成员距离损失 (member_distances)
    intra_group_loss = th.tensor(0.0, device=device, dtype=all_starts[0].dtype)
    num_intra_groups = 0
    
    for group_name, members in inter_graph.get('groups', {}).items():
        if len(members) != 2:
            continue
        idx1, idx2 = members
        constraints = inter_graph['distance_constraints'].get(group_name, {})
        member_dist_str = constraints.get('member_distances')
        
        if member_dist_str:
            lower, upper = parse_distance_constraint(member_dist_str)
            if lower is not None:
                distances = compute_distance_xz(roots[idx1], roots[idx2])
                loss = soft_range_loss(distances, lower, upper)
                intra_group_loss += loss
                num_intra_groups += 1
    
    if num_intra_groups > 0:
        # 适当提高intra_group_loss的权重，例如5倍
        intra_group_loss = intra_group_loss / num_intra_groups
        total_loss += weight * intra_group_loss * 5.0  # 增加5倍权重
        loss_dict['intra_group'] = (weight * intra_group_loss * 5.0).item() # 记录包含权重的损失
    
    # 2. Group间距离损失 (group_distances)
    inter_group_loss = th.tensor(0.0, device=device, dtype=all_starts[0].dtype)
    num_inter_group_pairs = 0
    
    group_list = list(inter_graph.get('groups', {}).items())
    for i, (group_name1, members1) in enumerate(group_list):
        for j, (group_name2, members2) in enumerate(group_list):
            if i >= j:
                continue
            
            # 获取两个group的group_distances约束
            constraints1 = inter_graph['distance_constraints'].get(group_name1, {})
            dist_str1 = constraints1.get('group_distances')
            
            constraints2 = inter_graph['distance_constraints'].get(group_name2, {})
            dist_str2 = constraints2.get('group_distances')
            
            # 解析约束
            lower1, upper1 = parse_distance_constraint(dist_str1) if dist_str1 else (None, None)
            lower2, upper2 = parse_distance_constraint(dist_str2) if dist_str2 else (None, None)
            
            # 合并约束（取交集：下限取大，上限取小）
            effective_lower = None
            if lower1 is not None and lower2 is not None:
                effective_lower = max(lower1, lower2)
            elif lower1 is not None:
                effective_lower = lower1
            elif lower2 is not None:
                effective_lower = lower2
                
            effective_upper = None
            if upper1 is not None and upper2 is not None:
                effective_upper = min(upper1, upper2)
            elif upper1 is not None:
                effective_upper = upper1
            elif upper2 is not None:
                effective_upper = upper2
            
            # 如果有有效约束，计算Loss
            if effective_lower is not None or effective_upper is not None:
                # 计算两组成员root中心位置点之间的距离
                # 计算group1的中心位置
                group1_roots = th.stack([roots[idx] for idx in members1], dim=0)  # (num_members1, B, T, 3)
                group1_center = group1_roots.mean(dim=0)  # (B, T, 3)
                
                # 计算group2的中心位置
                group2_roots = th.stack([roots[idx] for idx in members2], dim=0)  # (num_members2, B, T, 3)
                group2_center = group2_roots.mean(dim=0)  # (B, T, 3)
                
                # 计算两个中心点之间的距离
                center_distances = compute_distance_xz(group1_center, group2_center)
                center_loss = soft_range_loss(center_distances, effective_lower, effective_upper)
                
                # 额外增加对组间最近成员对的惩罚（确保任意两个成员都不太近）
                member_pair_loss = th.tensor(0.0, device=device, dtype=all_starts[0].dtype)
                num_pairs = 0
                for idx1 in members1:
                    for idx2 in members2:
                        member_dist = compute_distance_xz(roots[idx1], roots[idx2])
                        pair_loss = soft_range_loss(member_dist, effective_lower, effective_upper)
                        member_pair_loss += pair_loss
                        num_pairs += 1
                
                if num_pairs > 0:
                    member_pair_loss = member_pair_loss / num_pairs
                
                # 组间约束：中心点距离 + 成员对距离
                inter_group_loss += (center_loss + member_pair_loss) / 2.0
                num_inter_group_pairs += 1
    
    if num_inter_group_pairs > 0:
        inter_group_loss = inter_group_loss / num_inter_group_pairs
        total_loss += weight * inter_group_loss
        loss_dict['inter_group'] = inter_group_loss.item()
    
    # 3. Person与目标Group距离损失 (object_distances)
    # 轻微降权：Person->Group 距离项、以及朝向(Orientation)项
    # - person_to_group_dist_scale: 缩放距离项（中心点+成员对距离）
    # - orientation_scale: 缩放朝向项（1 - cos）
    person_to_group_dist_scale = 0.7
    orientation_scale = 0.3

    person_target_loss = th.tensor(0.0, device=device, dtype=all_starts[0].dtype)
    num_person_target_pairs = 0
    
    for person_idx, target_group_name in inter_graph.get('person_to_group', {}).items():
        target_members = inter_graph['groups'].get(target_group_name, [])
        person_id = inter_graph['idx_to_char'].get(person_idx, f'P{person_idx+1}')
        constraints = inter_graph['distance_constraints'].get(person_id, {})
        object_dist_str = constraints.get('object_distances')
        
        if object_dist_str and target_members:
            lower, upper = parse_distance_constraint(object_dist_str)
            if lower is not None:
                # 1. 计算两组成员root中心位置点之间的距离
                person_root = roots[person_idx]
                # 计算group的中心位置
                target_roots = th.stack([roots[idx] for idx in target_members], dim=0)
                target_center = target_roots.mean(dim=0)
                
                # 计算两个中心点之间的距离
                center_distances = compute_distance_xz(person_root, target_center)
                center_loss = soft_range_loss(center_distances, lower, upper)
                
                # 2. Person到成员root中心的向量与person头部朝向向量一致的约束
                person_joints = joints_list[person_idx]
                person_forward = compute_forward_direction(person_joints)  # (B, T, 2) 头部朝向
                to_target_dir = compute_direction_vector_xz(person_root, target_center)  # (B, T, 2)
                
                # Orientation loss: 直接使用角度（弧度），让小角度偏离也有足够惩罚
                # 0° → 0, 30° → 0.52, 45° → 0.79, 90° → 1.57, 180° → 3.14
                dot_prod = (person_forward * to_target_dir).sum(dim=-1)  # (B, T)
                # 使用 arccos 计算实际角度偏差（弧度）
                angle_rad = th.acos(th.clamp(dot_prod, -1.0, 1.0))  # [0, π]
                orientation_loss = angle_rad.mean()
                
                # 3. 额外增加对组间最近成员对的惩罚（确保任意两个成员都不太近）
                # Person与目标组所有成员的距离
                member_pair_loss = th.tensor(0.0, device=device, dtype=all_starts[0].dtype)
                num_pairs = 0
                
                for target_idx in target_members:
                    distances = compute_distance_xz(roots[person_idx], roots[target_idx])
                    loss = soft_range_loss(distances, lower, upper)
                    member_pair_loss += loss
                    num_pairs += 1
                
                if num_pairs > 0:
                    member_pair_loss = member_pair_loss / num_pairs
                
                # 4. 组间约束：中心点距离 + 成员对距离 + 朝向约束（分别降权）
                distance_term = (center_loss + member_pair_loss) / 2.0
                current_loss = person_to_group_dist_scale * distance_term + orientation_scale * orientation_loss
                
                person_target_loss += current_loss
                num_person_target_pairs += 1
    
    if num_person_target_pairs > 0:
        person_target_loss = person_target_loss / num_person_target_pairs
        total_loss += weight * person_target_loss
        loss_dict['person_target'] = person_target_loss.item()
    
    # 4. Person与其他Group距离损失 (group_distances from Person)
    person_other_loss = th.tensor(0.0, device=device, dtype=all_starts[0].dtype)
    num_person_other_pairs = 0
    
    for person_idx, target_group_name in inter_graph.get('person_to_group', {}).items():
        person_id = inter_graph['idx_to_char'].get(person_idx, f'P{person_idx+1}')
        constraints = inter_graph['distance_constraints'].get(person_id, {})
        group_dist_str = constraints.get('group_distances')  # 与其他组的距离
        
        if group_dist_str:
            lower, upper = parse_distance_constraint(group_dist_str)
            if lower is not None:
                # 遍历所有非目标组
                for group_name, members in inter_graph.get('groups', {}).items():
                    if group_name == target_group_name:
                        continue  # 跳过目标组（已在上面处理）
                    
                    # Person与其他组所有成员的距离
                    for other_idx in members:
                        distances = compute_distance_xz(roots[person_idx], roots[other_idx])
                        loss = soft_range_loss(distances, lower, upper)
                        person_other_loss += loss
                        num_person_other_pairs += 1
    
    if num_person_other_pairs > 0:
        person_other_loss = person_other_loss / num_person_other_pairs
        total_loss += weight * person_other_loss
        loss_dict['person_other'] = person_other_loss.item()
    
    # 5. Person与Person距离损失 (基于各自的group_distances约束)
    person_person_loss = th.tensor(0.0, device=device, dtype=all_starts[0].dtype)
    num_person_person_pairs = 0
    
    person_list = list(inter_graph.get('person_to_group', {}).items())
    for i, (person_idx1, target_group1) in enumerate(person_list):
        for j, (person_idx2, target_group2) in enumerate(person_list):
            if i >= j:
                continue  # 避免重复计算
            
            person_id1 = inter_graph['idx_to_char'].get(person_idx1, f'P{person_idx1+1}')
            person_id2 = inter_graph['idx_to_char'].get(person_idx2, f'P{person_idx2+1}')
            
            # 获取两个Person的group_distances约束
            constraints1 = inter_graph['distance_constraints'].get(person_id1, {})
            dist_str1 = constraints1.get('group_distances')
            
            constraints2 = inter_graph['distance_constraints'].get(person_id2, {})
            dist_str2 = constraints2.get('group_distances')
            
            # 解析约束
            lower1, upper1 = parse_distance_constraint(dist_str1) if dist_str1 else (None, None)
            lower2, upper2 = parse_distance_constraint(dist_str2) if dist_str2 else (None, None)
            
            # 合并约束（取交集：下限取大，上限取小）
            effective_lower = None
            if lower1 is not None and lower2 is not None:
                effective_lower = max(lower1, lower2)
            elif lower1 is not None:
                effective_lower = lower1
            elif lower2 is not None:
                effective_lower = lower2
                
            effective_upper = None
            if upper1 is not None and upper2 is not None:
                effective_upper = min(upper1, upper2)
            elif upper1 is not None:
                effective_upper = upper1
            elif upper2 is not None:
                effective_upper = upper2
            
            # 如果有有效约束，计算Loss
            if effective_lower is not None or effective_upper is not None:
                distances = compute_distance_xz(roots[person_idx1], roots[person_idx2])
                loss = soft_range_loss(distances, effective_lower, effective_upper)
                person_person_loss += loss
                num_person_person_pairs += 1
    
    if num_person_person_pairs > 0:
        person_person_loss = person_person_loss / num_person_person_pairs
        total_loss += weight * person_person_loss
        loss_dict['person_person'] = person_person_loss.item()
    
    return total_loss, loss_dict

