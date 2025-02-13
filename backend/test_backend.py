import sys
from deploy.utils import get_job_listings, get_explanations

def test_scenario(disability_vector, scenario_name):
    print(f"\n=== Testing Scenario: {scenario_name} ===")
    print(f"Disability Vector: {disability_vector}")
    
    # Get job recommendations
    job_listings = get_job_listings(disability_vector)
    
    # Get explanations
    explanations = get_explanations(disability_vector)
    
    # Print results
    print("\nRecommended Jobs:")
    for job in job_listings:
        print(f"- {job['job_title']} at {job['company']}")
    
    print("\nExplanations:")
    for explanation in explanations:
        print(f"- {explanation}")
    
    print("\n" + "="*50)

def run_tests():
    # Test Scenario 1: No disabilities
    test_scenario([0, 0, 0, 0, 0, 0, 0, 0], "No Disabilities")
    
    # Test Scenario 2: Visual impairment only
    test_scenario([1, 0, 0, 0, 0, 0, 0, 0], "Visual Impairment")
    
    # Test Scenario 3: Multiple disabilities
    test_scenario([1, 1, 1, 0, 0, 0, 0, 0], "Multiple Disabilities")
    
    # Test Scenario 4: All disabilities
    test_scenario([1, 1, 1, 1, 1, 1, 1, 1], "All Disabilities")
    
    # Test Scenario 5: Computer and creativity difficulties
    test_scenario([0, 0, 0, 0, 0, 0, 1, 1], "Computer and Creativity Difficulties")

if __name__ == "__main__":
    try:
        run_tests()
        print("\nAll test scenarios completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        sys.exit(1) 