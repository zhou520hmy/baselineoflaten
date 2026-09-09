from .prompts import INFORMATION_SYSTEM,INTEGRATOR_SYSTEM,VERIFIER_SYSTEM,information_questions

ROLE_INSTRUCTIONS={
    'information':'Preserve every assigned private status in the internal communication state for downstream roles. Emit no action label.',
    'integrator':'Integrate the accumulated internal state while preserving every status.',
    'verifier':'Verify and preserve the accumulated internal status record.',
}

ROLE_SYSTEMS={'information':INFORMATION_SYSTEM,'integrator':INTEGRATOR_SYSTEM,'verifier':VERIFIER_SYSTEM}

def build_role_public_context(case,program,role):
    """Public role context only; no variant argument, realized values or policy."""
    if role.role_type not in ROLE_SYSTEMS:raise ValueError('Only producer roles use this public context')
    return (
        {'role':'system','content':f'{ROLE_SYSTEMS[role.role_type]} Your role is {role.role_name}.'},
        {'role':'user','content':f'Domain: {case.domain}\nPublic scenario:\n{case.public_scenario}\n\n'
         'Public ordered fact definitions and binary code mapping:\n'+'\n'.join(information_questions(program))+
         '\n\nCurrent role instruction:\n'+ROLE_INSTRUCTIONS[role.role_type]},
    )
