from enum import Enum, auto


class CustomValueEnum(Enum):
    # Allows string values instead of numerical
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)

    @classmethod
    def factory(cls, name):
        for enum in cls:
            if name in {enum.name, enum.value}:
                return enum

    def __str__(self):
        return str(self.value)


class IntentBase(CustomValueEnum):
    # Values should match the Excel training file
    end_of_conversation = 'Conclusions'
    small_talk = 'Smalltalk'
    JOB_in_LOCATION = '[JOB]_in_[LOCATION]'
    skills_for_JOB = 'skills_for_[JOB]'

    processing_intents = {JOB_in_LOCATION, skills_for_JOB}

    def __init__(self, intent, processing_intents=(JOB_in_LOCATION, skills_for_JOB)):
        self.is_follow_up_valid = intent in processing_intents
        self.will_process = intent in processing_intents

    def get_result_intent(self):
        # Factory design pattern
        if self == IntentBase.JOB_in_LOCATION:
            return IntentResult_JOB_in_LOCATION
        elif self == IntentBase.skills_for_JOB:
            return IntentResult_skills_for_JOB

    def get_follow_up_intent(self):
        # Factory design pattern
        if self.is_follow_up_valid:
            return IntentFollowUp

    def get_requirements(self):
        # Returns a dictionary with keys as EntityBase Enums and values as the number of that entity required
        entity_requirements = EntityRequirements()
        if self == IntentBase.JOB_in_LOCATION:
            entity_requirements.add(EntityBase.J)
            entity_requirements.add(EntityBase.L)
        elif self == IntentBase.skills_for_JOB:
            entity_requirements.add(EntityBase.J)
        return entity_requirements


class IntentFollowUp(CustomValueEnum):
    # Values should match the Excel training file
    reject = 'Rejection'
    accept = 'Acceptance'


class IntentResult_JOB_in_LOCATION(Enum):
    description = auto()
    heatmap = auto()
    wordcloud = auto()
    table = auto()


class IntentResult_skills_for_JOB(Enum):
    # TODO: Fill this out when I get to it
    description = auto()


class StateBase(Enum):
    base = auto()
    seeking_additional_info = auto()
    ready_to_process = auto()
    processing = auto()
    conversation_complete = auto()


class EntityBase(CustomValueEnum):
    # Values should match the Excel training file
    J = 'job'
    L = 'location'


class BaseEntityDict(dict):
    def __init__(self):
        super().__init__()

    def add(self, entity, val):
        raise NotImplementedError

    def subtract(self, entity, val):
        raise NotImplementedError


class EntityRequirements(BaseEntityDict):
    def __init__(self):
        super().__init__()
        for entity in EntityBase:
            self[entity] = 0

    def add(self, entity, val=1):
        self[entity] += val

    def subtract(self, entity, val=1):
        self[entity] = max(0, self[entity] - val)

    def subtract_recognized_entities(self, recognized_entities):
        for entity, num_required in self.items():
            self.subtract(entity, len(recognized_entities[entity]))

    def is_satisfied(self):
        return self.size == 0

    @property
    def size(self):
        return sum([val for val in self.values()])

    def get_missing_entity(self):
        if not self.is_satisfied():
            for entity in self:
                return entity  # Return first entity found

    def __str__(self):
        output_str = ''
        for entity, num in self.items():
            if num > 0:
                if output_str != '':
                    output_str += ' / '
                output_str += f'{entity}: {num}'

        return output_str


class RecognizedEntities(BaseEntityDict):
    def __init__(self):
        super().__init__()
        for entity in EntityBase:
            self[entity] = []

    def add(self, entity, val):
        self[entity].append(val)

    def append_to_latest(self, entity, val):
        self[entity][-1] += ' ' + val

    def subtract(self, entity, val):
        if val not in self[entity]:
            return
        else:
            self[entity].remove(val)

    def __str__(self):
        output_str = ''
        for entity, recognized_entity_list in self.items():
            if len(recognized_entity_list):
                if output_str != '':
                    output_str += ' / '
                output_str += str(entity) + ': ' + ', '.join(str(entity) for entity in recognized_entity_list)

        return output_str

    def is_empty(self):
        return self.size == 0

    @property
    def size(self):
        return sum([len(val) for val in self.values()])
