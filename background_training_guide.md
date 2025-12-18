# Running Training Scripts in Background

## Option 1: Using `screen` (RECOMMENDED - Best for already running scripts)

If your script is already running:

1. **Detach the current session:**
   ```bash
   # Press: Ctrl+A, then D (detach)
   # Or if using screen:
   screen -d
   ```

2. **Reattach later:**
   ```bash
   screen -r
   # Or list sessions:
   screen -ls
   ```

3. **Save output from current session:**
   ```bash
   # In the screen session, you can scroll up and copy output
   # Or redirect to file before detaching:
   # Press Ctrl+A then : (colon), then type:
   hardcopy -h output.txt
   ```

## Option 2: Using `tmux` (Alternative)

1. **Detach:**
   ```bash
   # Press: Ctrl+B, then D
   ```

2. **Reattach:**
   ```bash
   tmux attach
   ```

## Option 3: Using `nohup` + `disown` (For new runs)

If you want to start a new run in background:

```bash
# Start script with nohup (outputs to nohup.out)
nohup python train_vit_advanced.py > training_output.log 2>&1 &

# Or if already running, use disown:
# 1. Press Ctrl+Z to suspend
# 2. Type: bg (brings to background)
# 3. Type: disown (detaches from terminal)
```

## Option 4: Save Current Output + Move to Background

If script is currently running in terminal:

1. **Save current terminal output:**
   ```bash
   # In another terminal, find the process:
   ps aux | grep train_vit_advanced.py
   
   # Or use script command to log everything:
   script training_session.log
   # (then run your command, exit when done)
   ```

2. **Move running process to background:**
   ```bash
   # Press Ctrl+Z (suspends process)
   # Then:
   bg                    # Resume in background
   disown                # Detach from terminal
   ```

## Option 5: Redirect Output to File (Best Practice)

For future runs, always redirect output:

```bash
# Start in background with output logging
nohup python train_vit_advanced.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Or with timestamps in output:
python train_vit_advanced.py 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log &
```

## Checking Progress

```bash
# View output file in real-time:
tail -f training_output.log

# Or with screen:
screen -r  # Then scroll to see output
```

## Stopping Background Process

```bash
# Find process:
ps aux | grep train_vit_advanced.py

# Kill gracefully:
kill <PID>

# Or force kill:
kill -9 <PID>
```



