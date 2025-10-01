from apscheduler.schedulers.blocking import BlockingScheduler
import os

def retrain():
    """Function to run the retrain_with_feedback.py script."""
    print("Starting retraining...")
    os.system("python e:\\amir\\classification_welodge\\retrain_with_feedback.py")
    print("Retraining completed.")

# Create a scheduler
scheduler = BlockingScheduler()

# Schedule the retrain function to run on the 1st of every month at midnight
scheduler.add_job(retrain, 'cron', day=1, hour=0, minute=0)

print("Scheduler is running. Retraining will occur on the 1st of every month at midnight.")
try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    print("Scheduler stopped.")