import time

class TrafficLogic:
    def __init__(self):
        self.lanes = {
            1: {'status': 'RED', 'density': 0, 'ambulance': False, 'remaining_time': 0, 'amb_start_time': None},
            2: {'status': 'RED', 'density': 0, 'ambulance': False, 'remaining_time': 0, 'amb_start_time': None},
            3: {'status': 'RED', 'density': 0, 'ambulance': False, 'remaining_time': 0, 'amb_start_time': None},
            4: {'status': 'RED', 'density': 0, 'ambulance': False, 'remaining_time': 0, 'amb_start_time': None}
        }
        
        self.current_green_lane = 1
        self.lanes[1]['status'] = 'GREEN'
        
        # Configuration
        self.min_green_time = 10    
        self.density_threshold = 20  # Trigger for the 20s rule
        self.high_density_time = 20  # Fixed 20s for high density
        self.orange_duration = 3    
        
        self.last_update_time = time.time()
        self.lanes[1]['remaining_time'] = self.min_green_time

    def update_state(self, lane_id, current_density, ambulance_present):
        self.lanes[lane_id]['density'] = current_density
        self.lanes[lane_id]['ambulance'] = ambulance_present

    def get_lane_status(self, lane_id):
        # 1. EMERGENCY CHECK: Priority 1
        for lid, data in self.lanes.items():
            if data['ambulance']:
                self._handle_ambulance(lid)
                return self.lanes[lane_id]['status']
            else:
                # Reset ambulance timer if it's gone
                self.lanes[lid]['amb_start_time'] = None

        # 2. NORMAL TIMER/DENSITY LOGIC
        self._manage_timers()
        return self.lanes[lane_id]['status']

    def _manage_timers(self):
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        self.last_update_time = current_time

        lane = self.lanes[self.current_green_lane]
        lane['remaining_time'] -= elapsed

        if lane['remaining_time'] <= 0:
            if lane['status'] == 'GREEN':
                lane['status'] = 'ORANGE'
                lane['remaining_time'] = self.orange_duration
            elif lane['status'] == 'ORANGE':
                lane['status'] = 'RED'
                lane['remaining_time'] = 0
                
                # Move to next lane
                self.current_green_lane = (self.current_green_lane % 4) + 1
                new_lane = self.lanes[self.current_green_lane]
                
                # Logic: Density >= 20 gets 20s, else min_green_time
                if new_lane['density'] >= self.density_threshold:
                    assigned_time = self.high_density_time
                else:
                    assigned_time = self.min_green_time
                
                new_lane['status'] = 'GREEN'
                new_lane['remaining_time'] = assigned_time

    def _handle_ambulance(self, priority_lane):
        """Forces GREEN and tracks how long ambulance is present"""
        current_time = time.time()
        
        for lid in self.lanes:
            if lid == priority_lane:
                self.lanes[lid]['status'] = 'GREEN'
                # Initialize start time to track duration
                if self.lanes[lid]['amb_start_time'] is None:
                    self.lanes[lid]['amb_start_time'] = current_time
                
                # Calculate elapsed time since detection for analysis
                self.lanes[lid]['remaining_time'] = current_time - self.lanes[lid]['amb_start_time']
            else:
                self.lanes[lid]['status'] = 'RED'
                self.lanes[lid]['remaining_time'] = 0
        
        self.current_green_lane = priority_lane

    def get_timer_text(self, lane_id):
        lane = self.lanes[lane_id]
        if lane['ambulance']:
            # Show "AMB: X.Xs" to analyze passing time
            return f"AMB: {lane['remaining_time']:.1f}s"
        
        t = lane['remaining_time']
        return f"{int(t)}s" if t > 0 else "0s"