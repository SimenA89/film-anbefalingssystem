def validate_login(username, password):
    """
    Validates the username and password.
    
    Args:
        username (str): The username entered by the user.
        password (str): The password entered by the user.
    
    Returns:
        bool: True if the username and password are valid, False otherwise.
    """
    # Example hardcoded credentials for validation
    valid_username = "admin"
    valid_password = "password123"
    
    return username == valid_username and password == valid_password

# Example usage
if __name__ == "__main__":
    print("Login Form")
    input_username = input("Enter username: ")
    input_password = input("Enter password: ")
    
    if validate_login(input_username, input_password):
        print("Login successful!")
    else:
        print("Invalid username or password.")