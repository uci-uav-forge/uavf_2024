class Payload():
    def __init__(self, payload_description):
        self.feature_dict = {}

        for feature in payload_description:
            key, value = feature.strip().split(':')
            self.feature_dict[key] = int(value)

        self.shape_color_id = self.feature_dict['shape color']
        self.shape_id = self.feature_dict['shape']
        self.letter_color_id = self.feature_dict['letter color']
        self.letter_id = self.feature_dict['letter']
    
    def display(self):
        print(f'Shape color: {self.shape_color_id}, Shape: {self.shape_id}, Letter color: {self.letter_color_id}, Letter: {self.letter_id}')