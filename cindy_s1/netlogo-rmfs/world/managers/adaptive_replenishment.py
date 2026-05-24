from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from world.entities.pod import Pod

@dataclass
class SKUAssessment:
    sku_id: str
    priority: int
    inv_ratio: float

@dataclass
class PodAssessment:
    threshold: float
    priority: int
    needs_replenishment: bool
    current_fill_level: float
    skus_to_replenish: List[SKUAssessment]

class AdaptiveReplenishmentPolicy:
    def __init__(self):
        # Define class weights as (threshold_weight, priority_weight)
        self.class_weights = {
            0: (0.6, 3),  # Class 0: High threshold, high priority (fast movers)
            1: (0.5, 2),  # Class 1: Medium-high threshold, medium-high priority
            2: (0.4, 1),  # Class 2: Medium threshold, medium priority
            3: (0.4, 4),  # Class 3: Medium-low threshold, medium-low priority
            4: (0.3, 5),  # Class 4: Low threshold, low priority
            5: (0.2, 6)   # Class 5: Very low threshold, very low priority (slow movers)
        }
    
    def get_skus_in_pod(self, pod: Pod) -> List[str]:
        """Get list of SKU IDs in the specified pod"""
        if pod is None:
            return []
        return list(pod.skus.keys())
    
    def calculate_average_inventory_in_pod(self, pod: Pod) -> float:
        """Calculate average inventory fill level in the pod"""
        if pod is None or not pod.skus:
            return 0.0
        
        total_ratio = 0.0
        count = 0
        
        for sku_id, sku_data in pod.skus.items():
            current_qty = float(sku_data.get('current_qty', 0))
            limit_qty = float(sku_data.get('limit_qty', 1))
            
            if limit_qty > 0:
                ratio = current_qty / limit_qty
                total_ratio += ratio
                count += 1
        
        return total_ratio / count if count > 0 else 0.0
    
    def get_current_inventory_level(self, sku_id: str, pod: Pod) -> float:
        """Get current inventory level for SKU in pod"""
        if pod is None or sku_id not in pod.skus:
            return 0.0
        return float(pod.skus[sku_id].get('current_qty', 0))
    
    def get_max_inventory_level(self, sku_id: str, pod: Pod) -> float:
        """Get maximum inventory level for SKU in pod"""
        if pod is None or sku_id not in pod.skus:
            return 1.0
        return float(pod.skus[sku_id].get('limit_qty', 1))
    
    def clip(self, value: float, min_val: float, max_val: float) -> float:
        """Clip value between min and max bounds"""
        return max(min_val, min(max_val, value))
    
    def process_pod(self, pod: Pod, item_dictionary: Dict) -> PodAssessment:
        """
        Process a single pod to determine replenishment needs
        Args:
            pod: The pod to assess
            item_dictionary: Dictionary containing SKU data including item classes/index_class_priority
        Returns:
            PodAssessment object with replenishment recommendations
        """
        # Get SKUs in pod
        sku_list = self.get_skus_in_pod(pod)
        
        # Handle empty pod
        if not sku_list:
            return PodAssessment(
                threshold=0.5, # default threshold
                priority=2, # default priority
                needs_replenishment=False,
                current_fill_level=0.0,
                skus_to_replenish=[]
            )
        
        # Initialize class counts
        class_counts = {cls: 0 for cls in self.class_weights.keys()}
        
        # Count SKUs by class
        for sku_id in sku_list: 
            if sku_id in item_dictionary:
                item_class = item_dictionary[sku_id].get('item_class', 2)
                if item_class not in self.class_weights:
                    print(f"Warning: Unknown item class '{item_class}' for SKU {sku_id}, defaulting to class 2")
                    item_class = 2
                class_counts[item_class] += 1
        
        # Calculate weighted averages
        total_count = 0
        thr_sum = 0.0
        pri_sum = 0.0
        
        for cls, cnt in class_counts.items():
            if cnt > 0:
                tw, pw = self.class_weights[cls]
                thr_sum += cnt * tw
                pri_sum += cnt * pw
                total_count += cnt
        
        # Compute pod metrics
        pod_threshold = thr_sum / total_count if total_count > 0 else 0.6
        base_pod_priority = pri_sum / total_count if total_count > 0 else 2
        
        # Clip threshold
        final_threshold = self.clip(pod_threshold, 0.4, 0.6)
        
        # Check current fill and replenishment need
        current_fill = self.calculate_average_inventory_in_pod(pod)
        needs_replenish = current_fill < final_threshold
        
        # Calculate priority with inventory adjustment
        if current_fill < 0.4:
            inv_adj = 2  # High adjustment for very low inventory
        elif current_fill < 0.6:
            inv_adj = 1  # Medium adjustment for low inventory
        else:
            inv_adj = 0  # No adjustment
        
        final_priority = min(5, round(base_pod_priority + inv_adj))
        
        # Determine SKUs to replenish
        skus_to_replenish = []
        
        if needs_replenish:
            for sku_id in sku_list:
                if sku_id in item_dictionary:
                    item_class = item_dictionary[sku_id].get('item_class', 2)
                    if item_class not in self.class_weights:
                        print(f"Warning: Unknown item class '{item_class}' for SKU {sku_id}, defaulting to class 2")
                        item_class = 2
                    
                    sku_thr = self.class_weights[item_class][0]  # threshold_weight
                    
                    current_lvl = self.get_current_inventory_level(sku_id, pod)
                    max_lvl = self.get_max_inventory_level(sku_id, pod)
                    
                    inv_ratio = current_lvl / max_lvl if max_lvl > 0 else 0.0
                    
                    if inv_ratio < sku_thr:
                        # Calculate SKU priority adjustment
                        if inv_ratio < 0.4:
                            sku_adj = 2  # High priority for very low inventory
                        elif inv_ratio < 0.6:
                            sku_adj = 1  # Medium priority for low inventory
                        else:
                            sku_adj = 0  # No adjustment
                        
                        base_sku_pri = self.class_weights[item_class][1]  # priority_weight
                        sku_priority = min(5, base_sku_pri + sku_adj)
                        
                        skus_to_replenish.append(SKUAssessment(
                            sku_id=sku_id,
                            priority=sku_priority,
                            inv_ratio=inv_ratio
                        ))
        
        # Sort SKUs by priority (descending order - higher priority first)
        skus_to_replenish.sort(key=lambda x: x.priority, reverse=True)
        
        return PodAssessment(
            threshold=final_threshold,
            priority=final_priority,
            needs_replenishment=needs_replenish,
            current_fill_level=current_fill,
            skus_to_replenish=skus_to_replenish
        )
