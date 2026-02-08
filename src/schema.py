# Schema for the MongoDB database

from pydantic import BaseModel


class Appointment(BaseModel):
    name: str
    date: str
    time: str
    phone: str
    notes: str

    def to_dict(self):
        return {
            "name": self.name,
            "date": self.date,
            "time": self.time,
            "phone": self.phone,
            "notes": self.notes,
        }
