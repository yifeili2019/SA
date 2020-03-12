Header header
std_msgs/Int8 id
std_msgs/Float32 battery
std_msgs/Bool armed
std_msgs/Float32 last_heartbeat
std_msgs/Float32MultiArray home_lla
std_msgs/String mode
std_msgs/String system_status

std_msgs/Float32MultiArray pos_lla
std_msgs/Float32MultiArray pos_ned
std_msgs/Float32MultiArray pos
std_msgs/Float32MultiArray attitude_ned
std_msgs/Float32MultiArray attitude
std_msgs/Float32 heading_ned
std_msgs/Float32 heading
std_msgs/Float32MultiArray velocity_ned
std_msgs/Float32MultiArray velocity
std_msgs/Float32MultiArray groundspeed

Action action

std_msgs/Bool enable_agent
std_msgs/Bool enable_ca
std_msgs/Bool enable_wp
std_msgs/Bool enable_release
std_msgs/Bool enable_RTH
std_msgs/Bool enable_LAND
std_msgs/Bool enable_manual

std_msgs/Bool clear_ca
std_msgs/Bool clear_wp
std_msgs/Bool clear_release

std_msgs/String mission_state

std_msgs/Float32 pos_est