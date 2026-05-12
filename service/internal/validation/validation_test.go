package validation

import (
	"testing"

	"github.com/mobster1425/go-ml-inference-service/service/internal/model"
)

func TestValidInputPasses(t *testing.T) {
	if err := ValidateCustomerInput(validInput()); err != nil {
		t.Fatalf("expected valid input, got %v", err)
	}
}

func TestInvalidCreditScoreFails(t *testing.T) {
	input := validInput()
	input.CreditScore = 250
	if err := ValidateCustomerInput(input); err == nil {
		t.Fatal("expected invalid credit score to fail")
	}
}

func TestInvalidGeographyFails(t *testing.T) {
	input := validInput()
	input.Geography = "Canada"
	if err := ValidateCustomerInput(input); err == nil {
		t.Fatal("expected invalid geography to fail")
	}
}

func TestInvalidBinaryFieldsFail(t *testing.T) {
	input := validInput()
	input.HasCrCard = 2
	if err := ValidateCustomerInput(input); err == nil {
		t.Fatal("expected invalid HasCrCard to fail")
	}

	input = validInput()
	input.IsActiveMember = -1
	if err := ValidateCustomerInput(input); err == nil {
		t.Fatal("expected invalid IsActiveMember to fail")
	}
}

func validInput() model.CustomerInput {
	return model.CustomerInput{
		CreditScore:     650,
		Geography:       "France",
		Gender:          "Female",
		Age:             42,
		Tenure:          5,
		Balance:         125000,
		NumOfProducts:   2,
		HasCrCard:       1,
		IsActiveMember:  1,
		EstimatedSalary: 78000,
	}
}
