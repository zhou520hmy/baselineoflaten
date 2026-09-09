import json,os,signal,sys,resource
payload=json.load(sys.stdin)
os.chdir('/tmp');resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
def timeout(*args):raise TimeoutError('Expanded test time limit')
signal.signal(signal.SIGALRM,timeout);signal.alarm(int(payload['timeout']))
try:exec(compile(payload['program'],'<solution_and_expanded_tests>','exec'),{})
except TimeoutError as e:print(str(e),file=sys.stderr);sys.exit(124)
except BaseException as e:print(type(e).__name__+': '+str(e)[:1000],file=sys.stderr);sys.exit(1)
finally:signal.alarm(0)
