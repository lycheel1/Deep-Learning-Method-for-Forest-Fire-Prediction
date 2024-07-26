from datetime import date, timedelta


# transform 1-365/366 to real date
def dayIndex_to_date(year, dayIndex):
    start_of_year = date(year, 1, 1)
    actual_date = start_of_year + timedelta(days=dayIndex - 1)  # Subtract 1 because daynum starts from 1
    return actual_date