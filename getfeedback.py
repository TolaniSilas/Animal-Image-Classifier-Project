import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart



msg = MIMEText("This is the body of the email. I love you Silas.")
msg["Subject"] = "Message to Silas."
msg["From"] = "osunbatolani@gmail.com"
msg["To"] = "ashlyjohn3606@gmail.com"



# Send the mail via SMTP.
with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login("osunbatolani@gmail.com", "Tolani123#")
    server.sendmail(msg["From"], msg["To"], msg.as_string())



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # Function to send email
# def send_email(feedback):
#     # Email configuration
#     sender_email = "your_email@example.com"
#     receiver_email = "recipient_email@example.com"
#     password = "your_email_password"

#     # Create message
#     message = MIMEMultipart()
#     message["From"] = sender_email
#     message["To"] = receiver_email
#     message["Subject"] = "Feedback from Animal Classifier Model"

#     # Add feedback to the email body
#     message.attach(MIMEText(feedback, "plain"))

#     # Connect to SMTP server and send email
#     with smtplib.SMTP("smtp.example.com", 587) as server:
#         server.starttls()
#         server.login(sender_email, password)
#         server.sendmail(sender_email, receiver_email, message.as_string())

# # Feedback Form Section
# st.sidebar.title("Feedback")
# feedback = st.sidebar.text_area("Please share your feedback here:", max_chars=500)
# if st.sidebar.button("Submit"):
#     # Process the feedback and send email
#     send_email(feedback)
#     st.sidebar.success("Thank you for your feedback! It has been submitted successfully.")
