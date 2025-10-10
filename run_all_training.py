import os
import sys
import subprocess
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def run_training_script(script_name, model_name):
    """Run a training script and handle any errors"""
    print(f"\n{'='*80}")
    print(f"STARTING {model_name} TRAINING")
    print(f"{'='*80}")
    print(f"Script: {script_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run the training script with real-time output
        result = subprocess.run([sys.executable, script_name], timeout=7200)  # 2 hour timeout
        
        if result.returncode == 0:
            print(f"\n[SUCCESS] {model_name} training completed successfully!")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        else:
            print(f"\n[FAILED] {model_name} training failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] {model_name} training timed out after 2 hours")
        return False
    except Exception as e:
        print(f"\n[ERROR] Error running {model_name} training: {e}")
        return False

def check_prerequisites():
    """Check if all required files and directories exist"""
    print("="*80)
    print("CHECKING PREREQUISITES")
    print("="*80)
    
    # Check if data directory exists
    if not os.path.exists("removed background"):
        print("[ERROR] 'removed background' directory not found!")
        print("Please run background_removal_all.py first to prepare the dataset.")
        return False
    
    # Check if training scripts exist
    training_scripts = [
        "train_vgg19.py",
        "train_resnet50.py", 
        "train_densenet121.py",
        "train_mobilenet.py"
    ]
    
    missing_scripts = []
    for script in training_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"[ERROR] Missing training scripts: {missing_scripts}")
        return False
    
    # Check if comparison script exists
    if not os.path.exists("model_comparison.py"):
        print("[ERROR] model_comparison.py not found!")
        return False
    
    print("[OK] All prerequisites met!")
    return True

def main():
    """Main function to run all training and comparison"""
    print("="*80)
    print("MEDICINAL PLANT LEAF CLASSIFICATION - COMPLETE TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n[ERROR] Prerequisites not met. Please fix the issues above and try again.")
        return
    
    # Define training sequence
    training_sequence = [
        ("train_vgg19.py", "VGG19"),
        ("train_resnet50.py", "ResNet50"),
        ("train_densenet121.py", "DenseNet121"),
        ("train_mobilenet.py", "MobileNet")
    ]
    
    # Track results
    training_results = {}
    start_time = time.time()
    
    # Run training for each model
    for script_name, model_name in training_sequence:
        print(f"\n{'='*80}")
        print(f"PROGRESS: {training_sequence.index((script_name, model_name)) + 1}/{len(training_sequence)}")
        print(f"{'='*80}")
        
        success = run_training_script(script_name, model_name)
        training_results[model_name] = success
        
        if success:
            print(f"[SUCCESS] {model_name} training completed successfully")
        else:
            print(f"[FAILED] {model_name} training failed - continuing with next model")
        
        # Add a small delay between models
        time.sleep(5)
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful_models = [model for model, success in training_results.items() if success]
    failed_models = [model for model, success in training_results.items() if not success]
    
    print(f"\nSuccessful models: {len(successful_models)}/{len(training_sequence)}")
    for model in successful_models:
        print(f"  [OK] {model}")
    
    if failed_models:
        print(f"\nFailed models: {len(failed_models)}/{len(training_sequence)}")
        for model in failed_models:
            print(f"  [X] {model}")
    
    # Run comparison if at least one model was successful
    if successful_models:
        print(f"\n{'='*80}")
        print("RUNNING MODEL COMPARISON")
        print(f"{'='*80}")
        
        try:
            result = subprocess.run([sys.executable, "model_comparison.py"], timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print("[SUCCESS] Model comparison completed successfully!")
                print("\n" + "="*80)
                print("FINAL SUMMARY")
                print("="*80)
                print("[COMPLETE] Training and comparison pipeline completed!")
                print(f"[RESULTS] Results available in:")
                print("  - Individual model results folders (vgg19_results/, resnet50_results/, etc.)")
                print("  - model_comparison_results/ (comprehensive comparison)")
                print("\n[FILES] Key files generated:")
                print("  - model_comparison_results/model_comparison_summary.csv")
                print("  - model_comparison_results/comparison_report.txt")
                print("  - model_comparison_results/*.png (visualization plots)")
            else:
                print("[FAILED] Model comparison failed!")
                
        except subprocess.TimeoutExpired:
            print("[TIMEOUT] Model comparison timed out")
        except Exception as e:
            print(f"[ERROR] Error running model comparison: {e}")
    else:
        print("\n[ERROR] No models trained successfully. Cannot run comparison.")
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED")
    print(f"{'='*80}")
    print("Thank you for using the Medicinal Plant Leaf Classification Pipeline!")

if __name__ == "__main__":
    main()
