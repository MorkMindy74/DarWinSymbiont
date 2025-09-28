#!/usr/bin/env python3
"""
DarWinSymbiont Backend API Testing Suite
Tests all API endpoints for the scientific research application
"""

import requests
import json
import time
import sys
import os
from datetime import datetime
from pathlib import Path

class DarWinSymbiontAPITester:
    def __init__(self, base_url="https://pdf-evolution.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        self.uploaded_files = []
        self.run_id = None

    def log_test(self, name, success, details="", response_data=None):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        result = {
            "test": name,
            "status": status,
            "success": success,
            "details": details,
            "response_data": response_data,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        print(f"{status} - {name}: {details}")
        return success

    def test_health_check(self):
        """Test basic API connectivity"""
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=10)
            success = response.status_code == 200
            return self.log_test(
                "API Health Check", 
                success,
                f"Status: {response.status_code}"
            )
        except Exception as e:
            return self.log_test("API Health Check", False, f"Error: {str(e)}")

    def test_upload_endpoint(self):
        """Test PDF upload functionality"""
        try:
            # Create a mock PDF file for testing
            test_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"
            
            files = {
                'files': ('test_paper.pdf', test_content, 'application/pdf')
            }
            
            response = requests.post(f"{self.api_url}/upload", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'files' in data and len(data['files']) > 0:
                    self.uploaded_files = data['files']
                    return self.log_test(
                        "PDF Upload", 
                        True,
                        f"Uploaded {len(data['files'])} files, Job ID: {data.get('jobId', 'N/A')}",
                        data
                    )
                else:
                    return self.log_test("PDF Upload", False, "No files in response")
            else:
                return self.log_test(
                    "PDF Upload", 
                    False, 
                    f"Status: {response.status_code}, Response: {response.text[:200]}"
                )
                
        except Exception as e:
            return self.log_test("PDF Upload", False, f"Error: {str(e)}")

    def test_studies_endpoint(self):
        """Test getting studies list"""
        try:
            response = requests.get(f"{self.api_url}/studies", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                studies_count = len(data.get('studies', []))
                return self.log_test(
                    "Get Studies", 
                    True,
                    f"Retrieved {studies_count} studies",
                    data
                )
            else:
                return self.log_test(
                    "Get Studies", 
                    False, 
                    f"Status: {response.status_code}"
                )
                
        except Exception as e:
            return self.log_test("Get Studies", False, f"Error: {str(e)}")

    def test_llm_summarize(self):
        """Test LLM summarization endpoint"""
        if not self.uploaded_files:
            return self.log_test("LLM Summarize", False, "No uploaded files to test with")
        
        try:
            study_ids = [file['id'] for file in self.uploaded_files]
            payload = {"studyIds": study_ids}
            
            response = requests.post(
                f"{self.api_url}/llm/summarize", 
                json=payload, 
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                summaries_count = len(data.get('summaries', []))
                return self.log_test(
                    "LLM Summarize", 
                    True,
                    f"Generated {summaries_count} summaries",
                    data
                )
            else:
                return self.log_test(
                    "LLM Summarize", 
                    False, 
                    f"Status: {response.status_code}, Response: {response.text[:200]}"
                )
                
        except Exception as e:
            return self.log_test("LLM Summarize", False, f"Error: {str(e)}")

    def test_llm_problem(self):
        """Test LLM problem analysis endpoint"""
        if not self.uploaded_files:
            return self.log_test("LLM Problem Analysis", False, "No uploaded files to test with")
        
        try:
            study_ids = [file['id'] for file in self.uploaded_files]
            payload = {"studyIds": study_ids}
            
            response = requests.post(
                f"{self.api_url}/llm/problem", 
                json=payload, 
                timeout=60
            )
            
            success = response.status_code == 200
            if success:
                data = response.json()
                problems_count = len(data.get('problems', []))
                details = f"Generated {problems_count} problem analyses"
            else:
                details = f"Status: {response.status_code}"
                data = None
                
            return self.log_test("LLM Problem Analysis", success, details, data)
                
        except Exception as e:
            return self.log_test("LLM Problem Analysis", False, f"Error: {str(e)}")

    def test_llm_compare(self):
        """Test LLM comparison endpoint"""
        if not self.uploaded_files:
            return self.log_test("LLM Compare", False, "No uploaded files to test with")
        
        try:
            study_ids = [file['id'] for file in self.uploaded_files]
            payload = {"studyIds": study_ids}
            
            response = requests.post(
                f"{self.api_url}/llm/compare", 
                json=payload, 
                timeout=60
            )
            
            success = response.status_code == 200
            if success:
                data = response.json()
                has_comparison = bool(data.get('comparison'))
                details = f"Generated comparison: {has_comparison}"
            else:
                details = f"Status: {response.status_code}"
                data = None
                
            return self.log_test("LLM Compare", success, details, data)
                
        except Exception as e:
            return self.log_test("LLM Compare", False, f"Error: {str(e)}")

    def test_llm_improve(self):
        """Test LLM improvement suggestions endpoint"""
        if not self.uploaded_files:
            return self.log_test("LLM Improve", False, "No uploaded files to test with")
        
        try:
            study_ids = [file['id'] for file in self.uploaded_files]
            payload = {
                "studyIds": study_ids,
                "context": "Test context for improvement suggestions"
            }
            
            response = requests.post(
                f"{self.api_url}/llm/improve", 
                json=payload, 
                timeout=60
            )
            
            success = response.status_code == 200
            if success:
                data = response.json()
                has_suggestions = bool(data.get('suggestions'))
                details = f"Generated suggestions: {has_suggestions}"
            else:
                details = f"Status: {response.status_code}"
                data = None
                
            return self.log_test("LLM Improve", success, details, data)
                
        except Exception as e:
            return self.log_test("LLM Improve", False, f"Error: {str(e)}")

    def test_dws_run(self):
        """Test DarWinSymbiont simulation run"""
        if not self.uploaded_files:
            return self.log_test("DWS Run", False, "No uploaded files to test with")
        
        try:
            study_ids = [file['id'] for file in self.uploaded_files]
            payload = {
                "params": {
                    "popSize": 20,
                    "mutationRate": 0.1,
                    "generations": 10,
                    "seed": 42,
                    "objective": "Test optimization objective"
                },
                "studyIds": study_ids
            }
            
            response = requests.post(
                f"{self.api_url}/dws/run", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.run_id = data.get('runId')
                return self.log_test(
                    "DWS Run", 
                    True,
                    f"Started simulation with run ID: {self.run_id}",
                    data
                )
            else:
                return self.log_test(
                    "DWS Run", 
                    False, 
                    f"Status: {response.status_code}, Response: {response.text[:200]}"
                )
                
        except Exception as e:
            return self.log_test("DWS Run", False, f"Error: {str(e)}")

    def test_dws_summary(self):
        """Test DWS run summary endpoint"""
        if not self.run_id:
            return self.log_test("DWS Summary", False, "No run ID available")
        
        try:
            # Wait a bit for simulation to process
            time.sleep(3)
            
            response = requests.get(f"{self.api_url}/dws/{self.run_id}/summary", timeout=10)
            
            success = response.status_code == 200
            if success:
                data = response.json()
                status = data.get('status', 'unknown')
                details = f"Run status: {status}"
            else:
                details = f"Status: {response.status_code}"
                data = None
                
            return self.log_test("DWS Summary", success, details, data)
                
        except Exception as e:
            return self.log_test("DWS Summary", False, f"Error: {str(e)}")

    def test_latex_generation(self):
        """Test LaTeX paper generation"""
        if not self.run_id or not self.uploaded_files:
            return self.log_test("LaTeX Generation", False, "No run ID or uploaded files available")
        
        try:
            study_ids = [file['id'] for file in self.uploaded_files]
            payload = {
                "runId": self.run_id,
                "studyIds": study_ids,
                "context": "Test LaTeX generation context"
            }
            
            response = requests.post(
                f"{self.api_url}/llm/latex", 
                json=payload, 
                timeout=90
            )
            
            if response.status_code == 200:
                data = response.json()
                has_latex = bool(data.get('latex'))
                latex_length = len(data.get('latex', ''))
                return self.log_test(
                    "LaTeX Generation", 
                    True,
                    f"Generated LaTeX: {has_latex}, Length: {latex_length} chars",
                    {"has_latex": has_latex, "length": latex_length}
                )
            else:
                return self.log_test(
                    "LaTeX Generation", 
                    False, 
                    f"Status: {response.status_code}, Response: {response.text[:200]}"
                )
                
        except Exception as e:
            return self.log_test("LaTeX Generation", False, f"Error: {str(e)}")

    def test_applications_generation(self):
        """Test business applications generation"""
        if not self.uploaded_files:
            return self.log_test("Applications Generation", False, "No uploaded files available")
        
        try:
            study_ids = [file['id'] for file in self.uploaded_files]
            payload = {
                "runId": self.run_id,  # Optional
                "studyIds": study_ids
            }
            
            response = requests.post(
                f"{self.api_url}/llm/applications", 
                json=payload, 
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                cards_count = len(data.get('cards', []))
                return self.log_test(
                    "Applications Generation", 
                    True,
                    f"Generated {cards_count} application cards",
                    data
                )
            else:
                return self.log_test(
                    "Applications Generation", 
                    False, 
                    f"Status: {response.status_code}, Response: {response.text[:200]}"
                )
                
        except Exception as e:
            return self.log_test("Applications Generation", False, f"Error: {str(e)}")

    def test_context_aware_business_proposals(self):
        """Test context-aware business proposals API (NEW FEATURE)"""
        if not self.uploaded_files:
            return self.log_test("Context-Aware Business Proposals", False, "No uploaded files available")
        
        try:
            # Prepare mock paper findings and simulation results
            paper_findings = {
                "summaries": [{"text": "Test summary of research paper"}],
                "problems": [{"text": "Test problem analysis"}],
                "improvements": "Test improvement suggestions",
                "comparison_insights": "Test comparison insights",
                "study_count": len(self.uploaded_files),
                "domains": [f"test_domain_{i}" for i in range(len(self.uploaded_files))]
            }
            
            simulation_results = {
                "best_fitness": 0.8547,
                "convergence_generation": 67,
                "total_generations": 100,
                "avg_fitness": 0.6838,
                "performance_verdict": "outperformed"
            }
            
            payload = {
                "paperFindings": paper_findings,
                "simulationResults": simulation_results,
                "constraints": {"maxCards": 8, "tone": "concise"}
            }
            
            response = requests.post(
                f"{self.api_url}/llm/business", 
                json=payload, 
                timeout=90
            )
            
            if response.status_code == 200:
                data = response.json()
                proposals_count = len(data.get('proposals', []))
                return self.log_test(
                    "Context-Aware Business Proposals", 
                    True,
                    f"Generated {proposals_count} context-aware business proposals",
                    data
                )
            else:
                return self.log_test(
                    "Context-Aware Business Proposals", 
                    False, 
                    f"Status: {response.status_code}, Response: {response.text[:200]}"
                )
                
        except Exception as e:
            return self.log_test("Context-Aware Business Proposals", False, f"Error: {str(e)}")

    def test_data_consistency_check(self):
        """Test data consistency checking API (NEW FEATURE)"""
        if not self.run_id:
            return self.log_test("Data Consistency Check", False, "No run ID available")
        
        try:
            # Wait a bit more for simulation to complete
            time.sleep(5)
            
            response = requests.get(
                f"{self.api_url}/consistency/check?runId={self.run_id}", 
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                is_consistent = data.get('consistent', False)
                inconsistencies_count = len(data.get('inconsistencies', []))
                threshold = data.get('threshold_percent', 0)
                
                return self.log_test(
                    "Data Consistency Check", 
                    True,
                    f"Consistency: {is_consistent}, Inconsistencies: {inconsistencies_count}, Threshold: {threshold}%",
                    data
                )
            else:
                return self.log_test(
                    "Data Consistency Check", 
                    False, 
                    f"Status: {response.status_code}, Response: {response.text[:200]}"
                )
                
        except Exception as e:
            return self.log_test("Data Consistency Check", False, f"Error: {str(e)}")

    def test_compare_performance(self):
        """Test performance comparison API (NEW FEATURE)"""
        if not self.run_id or not self.uploaded_files:
            return self.log_test("Performance Comparison", False, "No run ID or uploaded files available")
        
        try:
            study_ids = [file['id'] for file in self.uploaded_files]
            payload = {
                "runId": self.run_id,
                "studyIds": study_ids,
                "context": "Test performance comparison context"
            }
            
            response = requests.post(
                f"{self.api_url}/llm/compare-performance", 
                json=payload, 
                timeout=90
            )
            
            if response.status_code == 200:
                data = response.json()
                verdict = data.get('verdict', 'unknown')
                has_summary = bool(data.get('summary'))
                comparison_table_count = len(data.get('comparisonTable', []))
                
                return self.log_test(
                    "Performance Comparison", 
                    True,
                    f"Verdict: {verdict}, Has summary: {has_summary}, Comparison entries: {comparison_table_count}",
                    data
                )
            else:
                return self.log_test(
                    "Performance Comparison", 
                    False, 
                    f"Status: {response.status_code}, Response: {response.text[:200]}"
                )
                
        except Exception as e:
            return self.log_test("Performance Comparison", False, f"Error: {str(e)}")

    def run_all_tests(self):
        """Run all API tests in sequence"""
        print("🧪 Starting DarWinSymbiont Backend API Tests")
        print(f"🌐 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Basic connectivity
        self.test_health_check()
        
        # Core functionality tests
        self.test_upload_endpoint()
        time.sleep(2)  # Allow processing time
        
        self.test_studies_endpoint()
        
        # LLM analysis tests
        self.test_llm_summarize()
        self.test_llm_problem()
        self.test_llm_compare()
        self.test_llm_improve()
        
        # Simulation tests
        self.test_dws_run()
        self.test_dws_summary()
        
        # Generation tests
        self.test_latex_generation()
        self.test_applications_generation()
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        print(f"✅ Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        # Print failed tests
        failed_tests = [r for r in self.test_results if not r['success']]
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['details']}")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    tester = DarWinSymbiontAPITester()
    
    try:
        success = tester.run_all_tests()
        
        # Save detailed results
        results_file = "/app/test_reports/backend_api_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": tester.tests_run,
                    "passed_tests": tester.tests_passed,
                    "success_rate": (tester.tests_passed/tester.tests_run)*100 if tester.tests_run > 0 else 0,
                    "timestamp": datetime.now().isoformat()
                },
                "test_results": tester.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())