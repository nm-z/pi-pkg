import serial, time, sys, glob

PORTS = sorted(glob.glob('/dev/ttyACM*'))
print("Candidates:", PORTS)

for dev in PORTS:
    try:
        print(f"\n=== {dev} ===")
        ser = serial.Serial(
            dev,
            115200,
            timeout=1,
            dsrdtr=False,   # leave DTR as-is
            rtscts=False,   # no hardware flow control
            write_timeout=1
        )

        time.sleep(4)      # mimic IDE’s settle time
        ser.reset_input_buffer()  # discard boot noise

        for i in range(3):
            ser.write(b'TEMP\r\n')  # CR+LF like the IDE
            ser.flush()
            line = ser.read_until(b'\n').decode('utf-8', 'ignore').strip()
            print(f"[{i}] {line or '(no data)'}")
            time.sleep(0.3)

    except serial.SerialException as e:
        print("SerialException:", e)
    finally:
        ser.close()
