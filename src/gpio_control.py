import RPi.GPIO as GPIO
import time
import threading
from utils import RED, GREEN, BLUE, YELLOW, RESET

GPIO_PIN = 40
GPIO_OUTPUT_PIN = 38
gpio_lock = threading.Lock()

def init_gpio():
    try:
        print(f"{BLUE}Setting GPIO mode...{RESET}")
        GPIO.setmode(GPIO.BOARD)
        print(f"{BLUE}Configuring PIN40...{RESET}")
        GPIO.setup(GPIO_PIN, GPIO.IN)
        print(f"{BLUE}Configuring PIN38...{RESET}")
        GPIO.setup(GPIO_OUTPUT_PIN, GPIO.OUT)
        GPIO.output(GPIO_OUTPUT_PIN, GPIO.LOW)
        print(f"{GREEN}GPIO initialization successful{RESET}")
        return True
    except Exception as e:
        print(f"{RED}GPIO initialization failed: {str(e)}{RESET}")
        print(f"{RED}Error details: {type(e).__name__}{RESET}")
        return False

def gpio_monitor_thread(fall_event_log, stop_event):
    print(f"{BLUE}GPIO monitoring thread started{RESET}")
    print(f"{BLUE}Starting to monitor PIN40 state and control PIN38 output...{RESET}")
    
    last_clear_time = 0
    clear_cooldown = 0.5
    
    last_state = False
    last_fall_status = False
    
    while not stop_event.is_set():
        try:
            raw_value = GPIO.input(GPIO_PIN)
            current_state = raw_value == GPIO.HIGH
            
            current_fall_status = len(fall_event_log) > 0
            
            if current_fall_status != last_fall_status:
                if current_fall_status:
                    GPIO.output(GPIO_OUTPUT_PIN, GPIO.HIGH)
                    print(f"{RED}PIN38: HIGH - Fall event detected{RESET}")
                else:
                    GPIO.output(GPIO_OUTPUT_PIN, GPIO.LOW)
                    print(f"{GREEN}PIN38: LOW - No fall event{RESET}")
                last_fall_status = current_fall_status
            
            if current_state != last_state:
                if current_state:
                    print(f"{YELLOW}Button pressed detected!{RESET}")
                    current_time = time.perf_counter()
                    if (current_time - last_clear_time) >= clear_cooldown:
                        with gpio_lock:
                            fall_event_log.clear()
                            print(f"{GREEN}Fall event log cleared{RESET}")
                            GPIO.output(GPIO_OUTPUT_PIN, GPIO.LOW)
                            print(f"{GREEN}PIN38: LOW - Fall event log cleared{RESET}")
                            last_fall_status = False
                        last_clear_time = current_time
                else:
                    print(f"{BLUE}Button released{RESET}")
                
                last_state = current_state
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"{RED}GPIO monitoring error: {str(e)}{RESET}")
            print(f"{RED}Error details: {type(e).__name__}{RESET}")
            time.sleep(1) 