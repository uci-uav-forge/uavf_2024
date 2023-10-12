from .imaging_types import TargetDescription, Target3D
def calc_match_score(target_desc: TargetDescription, target: Target3D):
        shape_score = target.shape_probs[target_desc.shape]
        letter_score = target.letter_probs[target_desc.letter]
        shape_color_score = target.shape_col_probs[target_desc.shape_color]
        letter_color_score = target.letter_col_probs[target_desc.letter_color]

        return shape_score * letter_score * shape_color_score * letter_color_score
