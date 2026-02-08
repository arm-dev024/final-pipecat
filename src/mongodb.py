from pymongo import MongoClient
from dotenv import load_dotenv
import os

from src.schema import Appointment

load_dotenv(override=True)


class MongoDB:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGO_URI"))
        self.db = self.client[os.getenv("MONGO_DB_NAME")]

    def run(self):
        print("--------------------------------")
        print("MongoDB is running")
        print("--------------------------------")

    def check_appointment_conflict(self, appointment_date: str, appointment_time: str):
        appointments = self.db["appointments"].find({})

        for appointment in appointments:
            if (
                appointment["date"] == appointment_date
                and appointment["time"] == appointment_time
            ):
                return True
        return False

    def insert_appointment(self, appointment: Appointment):
        if self.check_appointment_conflict(appointment.date, appointment.time):
            print("Appointment conflict found")
            return "Appointment conflict found. Choose a different date or time."
        self.db["appointments"].insert_one(appointment.to_dict())
        return "Appointment created successfully"


mongo_db = MongoDB()
