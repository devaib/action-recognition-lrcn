import re


class Anno(object):
    """
    Parameters:

        person_num: str
            number of the person;
            training:   11 - 18,
            validation: 19 - 25, 01, 04,
            testing:    22, 02, 03, 05 - 10
        action: str
            name of of action
        condition: str
            one of four conditions
        frames: dict
            dictionary for frames to different conditions
    """
    def __init__(self, person_num, action, condition, frames=None):
        self.person_num = person_num
        self.action = action
        self.condition = condition
        self.frames = []

    def get_condition(self):
        return self.condition

    def get_person_num(self):
        return self.person_num

    def get_videoname(self):
        return 'person' + str(self.person_num) + '_' + self.action + '_' + self.condition + '_uncomp'

    def get_frames(self):
        return self.frames

    def add_frames(self, add_frames):
        self.frames.append(add_frames)


class Annos(object):
    """
    list of Annotations
    Parameters:
    """
    def __init__(self):
        self.size = 0
        self.annos = []

    def add_anno(self, anno):
        self.annos.append(anno)

    def get_anno(self, person_num, action, condition):
        for anno in self.annos:
            if anno.person_num == person_num and anno.action == action and anno.condition == condition:
                return anno
        return None

    def get_anno_by_videoname(self, videoname):
        words = videoname.split('_')
        p_num = re.compile('\d+')   # pattern
        m_num = p_num.findall(words[0])[0]
        return self.get_anno(m_num, words[1], words[2])



