def get_recommendations(
        attendance,
        study_hours,
        assignment_score,
        previous_score,
        predicted_score
):
    recommendations=[]
    if attendance<75:
        recommendations.append(
            "Improve attendance and attend classes regularaly"
        )
    if study_hours<2:
        recommendations.append(
            "Create daily study schedule and increase focussed study time"
        )
    if assignment_score<60:
        recommendations.append(
            "Revice previous lession and solve practice quation"
        )
    if previous_score<50:
        recommendations.append(
            "revise previous lession and solve quation"
        )

    if predicted_score<40:
        recommendations.append(
            "Follow a stuctured revision plan and seek help from teacher"
        )
    elif predicted_score<70:
        recommendations.append(
            "practice more quation and take weekly mock"
        )
    else:
        recommendations.append(
            "Maintain consistant study habits and prectice advanced quation"
        )
    if not recommendations:
        recommendations.append(
            "keep following your study rutine and moniter progess"
        )
    return recommendations