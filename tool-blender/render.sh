MODE="verbose"
for var in "$@"
do
    if [ "$var" == "-Q" ] || [ "$var" == "--quiet" ] ; then
        MODE="quiet"
    fi
done

if [ "$MODE" == "verbose" ] ; then
    blender render_flame.blend --python render.py --background --render-anim -- "$@"
else
    blender render_flame.blend --python render.py --background --render-anim -- "$@"  > null
fi
