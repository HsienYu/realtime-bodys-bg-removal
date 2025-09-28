#!/usr/bin/env python3
"""
Configuration optimizer to help find the best settings for your specific use case
"""
import os

def suggest_optimal_settings():
    """Suggest optimal settings based on the observed issues"""
    print("=== Segmentation Quality Optimizer ===\n")
    
    print("Based on your screenshot showing incomplete background replacement,")
    print("here are the recommended settings to try:\n")
    
    print("🎯 **IMMEDIATE FIXES TO TRY:**")
    print("1. **Lower Confidence Threshold**: Try 0.3 instead of 0.5")
    print("   - This will detect persons with lower confidence")
    print("   - Should capture more of the person's outline")
    print()
    
    print("2. **Use a Better Model**: Try yolov8m-seg.pt instead of yolov8n-seg.pt")
    print("   - More accurate segmentation")
    print("   - Better edge detection")
    print()
    
    print("3. **Adjust Feathering**: Try reducing feather amount to 5-7")
    print("   - Less aggressive edge softening")
    print("   - Preserve more person pixels")
    print()
    
    print("4. **Enable Debug Mode**: Set debug visualization to 1")
    print("   - See exactly what the mask looks like")
    print("   - Identify specific problem areas")
    print()
    
    print("📋 **RECOMMENDED CONFIGURATION:**")
    print("When prompted in app_enhanced.py, use these values:")
    print("- Model: 2 (yolov8m-seg.pt)")
    print("- Confidence: 0.3") 
    print("- Feather: 7")
    print("- Debug: 1")
    print("- Background: 0 (Green)")
    print()
    
    print("🔧 **ADVANCED TUNING:**")
    print("If the above doesn't work, try these combinations:")
    print()
    
    configs = [
        {
            "name": "High Sensitivity",
            "model": "yolov8m-seg.pt", 
            "conf": "0.25",
            "feather": "5",
            "desc": "Catches more person pixels, minimal feathering"
        },
        {
            "name": "Balanced Quality", 
            "model": "yolov8l-seg.pt",
            "conf": "0.35", 
            "feather": "8",
            "desc": "Good accuracy with smooth edges"
        },
        {
            "name": "Maximum Accuracy",
            "model": "yolov8x-seg.pt",
            "conf": "0.4",
            "feather": "6", 
            "desc": "Best possible segmentation (slower)"
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"{i}. **{config['name']}**:")
        print(f"   - Model: {config['model']}")
        print(f"   - Confidence: {config['conf']}")
        print(f"   - Feather: {config['feather']}")
        print(f"   - Use case: {config['desc']}")
        print()
    
    print("🏃 **QUICK TEST PROCEDURE:**")
    print("1. Run: python app_enhanced.py")
    print("2. Choose model 2 (yolov8m-seg.pt)")
    print("3. Enter confidence: 0.3")
    print("4. Enter feather: 7") 
    print("5. Enable debug: 1")
    print("6. Choose background mode 0 (green)")
    print("7. Press 'H' to show the overlay info")
    print("8. Check if more of your body is green-screened")
    print()
    
    print("🐛 **TROUBLESHOOTING:**")
    print("- If still incomplete: Lower confidence to 0.25")
    print("- If edges are too sharp: Increase feather to 10-12") 
    print("- If detection is unstable: Use yolov8l-seg.pt or yolov8x-seg.pt")
    print("- If performance is slow: Keep yolov8n-seg.pt but lower confidence")
    print()
    
    print("📊 **WHAT THE DEBUG MODE SHOWS:**")
    print("- Enable debug mode (1) to see mask visualization")
    print("- Yellow text 'MASK DEBUG MODE' confirms it's active")
    print("- Check if the person detection covers your whole body")
    print("- Green areas should be ONLY the background, not the person")

def create_quick_config_file():
    """Create a config file with optimal settings"""
    config_content = """# Optimal Configuration for app_enhanced.py
# Based on segmentation quality analysis

# Model Selection (when prompted "Choose a model:")
# 0: yolov8n-seg.pt (fastest, lower accuracy)
# 1: yolov8s-seg.pt 
# 2: yolov8m-seg.pt (RECOMMENDED - good balance)
# 3: yolov8l-seg.pt (high accuracy)
# 4: yolov8x-seg.pt (highest accuracy, slowest)
MODEL_CHOICE = 2

# Detection confidence (when prompted "Enter detection confidence threshold")
# Lower = catches more person pixels but may have false positives
# Higher = more precise but may miss parts of the person
CONFIDENCE = 0.3

# Edge feathering (when prompted "Enter edge feathering amount")  
# 0 = sharp edges
# 5-10 = good balance
# 15-20 = very soft edges
FEATHER_AMOUNT = 7

# Debug visualization (when prompted "Enable mask debug visualization")
# 0 = no debug
# 1 = show debug info (RECOMMENDED for troubleshooting)
DEBUG_MODE = 1

# Background mode (when prompted "Choose background mode")
# 0 = Green background (RECOMMENDED for testing)
BACKGROUND_MODE = 0
"""
    
    with open("optimal_config.txt", "w") as f:
        f.write(config_content)
    
    print(f"📄 Created 'optimal_config.txt' with recommended settings")

if __name__ == "__main__":
    suggest_optimal_settings()
    create_quick_config_file()
    
    print("\n" + "="*60)
    print("🎉 NEXT STEPS:")
    print("1. Run: python app_enhanced.py")
    print("2. Follow the recommended configuration above")
    print("3. If issues persist, run: python debug_mask_quality.py")
    print("4. Report back what you see!")
    print("="*60)