def generate_recommendations(input_data):
    """Generates personalized recommendations based on student input."""
    recommendations = []
    
    hours = input_data['hours_studied']
    attendance = input_data['attendance_percentage']
    stress = input_data['stress_level']

    if hours < 5:
        recommendations.append("Increase weekly study time to at least 5-7 hours to improve understanding.")
    
    if attendance < 80:
        recommendations.append("Improving attendance can significantly boost performance. Try not to miss classes.")

    if stress == 'High':
        recommendations.append("High stress can negatively impact scores. Consider stress-management techniques or counseling support.")
        
    if not recommendations:
        recommendations.append("Keep up the great work! Your habits are on the right track.")

    return recommendations