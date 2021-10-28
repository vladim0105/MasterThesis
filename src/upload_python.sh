rsync -avm -e 'ssh -p 60441' --include='*/' --include='*.py' --exclude='*'\
  ~/Desktop/dev/MasterThesis/src/ vladim0105@dnat.simula.no:'MasterThesis/src/'