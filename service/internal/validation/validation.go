package validation

import (
	"errors"
	"fmt"

	"github.com/mobster1425/go-ml-inference-service/service/internal/model"
)

func ValidateCustomerInput(input model.CustomerInput) error {
	if input.CreditScore < 300 || input.CreditScore > 900 {
		return fmt.Errorf("CreditScore must be between 300 and 900")
	}
	if input.Age < 18 || input.Age > 100 {
		return fmt.Errorf("Age must be between 18 and 100")
	}
	if input.Tenure < 0 || input.Tenure > 10 {
		return fmt.Errorf("Tenure must be between 0 and 10")
	}
	if input.Balance < 0 {
		return fmt.Errorf("Balance must be greater than or equal to 0")
	}
	if input.NumOfProducts < 1 || input.NumOfProducts > 4 {
		return fmt.Errorf("NumOfProducts must be between 1 and 4")
	}
	if input.HasCrCard != 0 && input.HasCrCard != 1 {
		return fmt.Errorf("HasCrCard must be 0 or 1")
	}
	if input.IsActiveMember != 0 && input.IsActiveMember != 1 {
		return fmt.Errorf("IsActiveMember must be 0 or 1")
	}
	if input.EstimatedSalary < 0 {
		return fmt.Errorf("EstimatedSalary must be greater than or equal to 0")
	}
	if !validGeography(input.Geography) {
		return fmt.Errorf("Geography must be one of France, Germany, Spain")
	}
	if !validGender(input.Gender) {
		return fmt.Errorf("Gender must be Female or Male")
	}
	return nil
}

func ValidateBatch(inputs []model.CustomerInput) error {
	if len(inputs) == 0 {
		return errors.New("records must contain at least one customer")
	}
	for i, input := range inputs {
		if err := ValidateCustomerInput(input); err != nil {
			return fmt.Errorf("records[%d]: %w", i, err)
		}
	}
	return nil
}

func validGeography(value string) bool {
	return value == "France" || value == "Germany" || value == "Spain"
}

func validGender(value string) bool {
	return value == "Female" || value == "Male"
}
