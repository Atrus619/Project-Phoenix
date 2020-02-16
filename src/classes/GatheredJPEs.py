class GatheredJPEs:
    __slots__ = 'required_years_experience', 'required_degree', 'travel_percentage', 'salary', \
                'extracted_required_years_experience', 'extracted_required_degrees', 'extracted_travel_percentages', 'extracted_salaries'

    def __init__(self, required_years_experience=0, required_degree=0, travel_percentage=0, salary=0):
        self.required_years_experience = required_years_experience
        self.required_degree = required_degree
        self.travel_percentage = travel_percentage
        self.salary = salary

        self.extracted_required_years_experience, self.extracted_required_degrees, self.extracted_travel_percentages, self.extracted_salaries = [], [], [], []

    def is_complete(self):
        return self.required_years_experience == self.required_degree == self.travel_percentage == self.salary == 0

    def update(self, jpes):
        for jpe in jpes:
            extracted_years_experience = jpe.extract_required_years_experience()
            if extracted_years_experience is not None:
                self.required_years_experience -= 1
                self.extracted_required_years_experience.append(extracted_years_experience)

            extracted_required_degree = jpe.extract_required_degree()
            if extracted_required_degree is not None:
                self.required_degree -= 1
                self.extracted_required_degrees.append(extracted_required_degree)

            extracted_travel_percentage = jpe.extract_travel_percentage()
            if extracted_travel_percentage is not None:
                self.required_degree -= 1
                self.extracted_travel_percentages.append(extracted_travel_percentage)

            extracted_salary = jpe.extract_salary()
            if extracted_salary is not None:
                self.salary -= 1
                self.extracted_salaries.append(extracted_salary)
