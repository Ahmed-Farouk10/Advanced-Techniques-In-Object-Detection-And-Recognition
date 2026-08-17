import language_tool_python
import re

def clean_latex(text):
    # Remove latex commands for better grammar checking
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    return text

def check_grammar():
    print("Loading LanguageTool...")
    tool = language_tool_python.LanguageTool('en-US')
    
    with open('cognitive_fire_defense.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract only document body
    if '\\begin{document}' in content:
        content = content.split('\\begin{document}')[1]
    
    clean_text = clean_latex(content)
    
    print("Checking grammar...")
    matches = tool.check(clean_text)
    
    if len(matches) == 0:
        print("No grammar issues found!")
    else:
        print(f"Found {len(matches)} issues:")
        for match in matches:
            print(f"Line {match.offsetInContext}: {match.ruleIssueType} - {match.message}")
            print(f"Context: {match.context}")
            print("---")

if __name__ == "__main__":
    check_grammar()
