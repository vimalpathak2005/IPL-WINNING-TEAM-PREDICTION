import streamlit as st
import pickle
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    with open('pipe.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Team names (same as used in training)
teams = [
    'Sunrisers Hyderabad', 
    'Mumbai Indians', 
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 
    'Kings XI Punjab',
    'Chennai Super Kings', 
    'Rajasthan Royals',
    'Delhi Capitals'
]

# City names (from your data)
cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

def main():
    st.set_page_config(
        page_title="IPL Win Probability Predictor",
        page_icon="🏏",
        layout="wide"
    )
    
    # Title and description
    st.title("🏏 IPL Win Probability Predictor")
    st.markdown("""
    This app predicts the probability of a team winning in an IPL match during the second innings 
    based on the current match situation.
    """)
    
    # Load model
    try:
        model = load_model()
        st.sidebar.success("Model loaded successfully!")
    except:
        st.error("Error loading model. Please make sure 'pipe.pkl' is in the same directory.")
        return
    
    # Sidebar for user inputs
    st.sidebar.header("Match Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Team Information")
        batting_team = st.selectbox("Batting Team (Chasing)", teams)
        bowling_team = st.selectbox("Bowling Team (Defending)", 
                                   [team for team in teams if team != batting_team])
        city = st.selectbox("Venue City", cities)
    
    with col2:
        st.subheader("Current Match Situation")
        
        # Match statistics
        total_runs = st.number_input("Target Runs", min_value=1, max_value=300, value=150)
        current_score = st.number_input("Current Score", min_value=0, max_value=total_runs-1, value=50)
        overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=19.5, value=10.0, step=0.1)
        wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=9, value=2)
    
    # Calculate derived features
    if overs_completed > 0:
        runs_left = total_runs - current_score
        balls_left = 120 - int(overs_completed * 6) - (overs_completed % 1 * 10)
        wicket_left = 10 - wickets_lost
        
        # Avoid division by zero
        if (120 - balls_left) > 0:
            crr = (current_score * 6) / (120 - balls_left)
        else:
            crr = 0
            
        if balls_left > 0:
            rrr = (runs_left * 6) / balls_left
        else:
            rrr = 0
        
        # Display current match situation
        st.subheader("Current Match Situation")
        sit_col1, sit_col2, sit_col3, sit_col4 = st.columns(4)
        
        with sit_col1:
            st.metric("Runs Required", runs_left)
        with sit_col2:
            st.metric("Balls Remaining", balls_left)
        with sit_col3:
            st.metric("Wickets in Hand", wicket_left)
        with sit_col4:
            st.metric("Required Run Rate", f"{rrr:.2f}")
        
        # Create input dataframe for prediction
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wicket_left': [wicket_left],
            'total_runs_x': [total_runs],
            'crr': [crr],
            'rrr': [rrr]
        })
        
        # Make prediction
        if st.button("Predict Win Probability", type="primary"):
            try:
                probability = model.predict_proba(input_df)[0]
                win_prob = probability[1] * 100  # Probability of winning
                loss_prob = probability[0] * 100  # Probability of losing
                
                # Display results
                st.subheader("Prediction Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.metric(
                        f"{batting_team} Win Probability", 
                        f"{win_prob:.1f}%",
                        delta=f"+{win_prob:.1f}%" if win_prob > 50 else f"{win_prob:.1f}%",
                        delta_color="normal" if win_prob > 50 else "inverse"
                    )
                
                with result_col2:
                    st.metric(
                        f"{bowling_team} Win Probability", 
                        f"{loss_prob:.1f}%",
                        delta=f"+{loss_prob:.1f}%" if loss_prob > 50 else f"{loss_prob:.1f}%",
                        delta_color="normal" if loss_prob > 50 else "inverse"
                    )
                
                # Progress bars for visualization
                st.progress(win_prob/100, text=f"{batting_team} winning chance")
                st.progress(loss_prob/100, text=f"{bowling_team} winning chance")
                
                # Additional insights
                if win_prob > 70:
                    st.success(f"🚀 {batting_team} is in a strong position to win!")
                elif win_prob > 50:
                    st.info(f"⚖️ The match is balanced, slight edge to {batting_team}")
                else:
                    st.warning(f"🏏 {bowling_team} has the upper hand in this situation")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    else:
        st.info("Please enter the current match situation to get predictions.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note**: This model predicts win probability based on historical IPL data. "
        "Actual match outcomes may vary due to various factors."
    )

if __name__ == "__main__":
    main()