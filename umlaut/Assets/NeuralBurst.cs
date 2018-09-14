using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

#if UNITY_EDITOR
using UnityEditor;

[CustomEditor(typeof(NeuralBurst))]
public class NeuralBurstEditor
    : Editor
{
    private bool showLayer0;
    private bool showLayer1;
    private bool showLayer2;

    private bool showWeights10;
    private bool showWeights21;

    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        if (!Application.isPlaying)
            return;

        var self = target as NeuralBurst;

        if (GUILayout.Button("Configure 1-1-1"))
            self.Configure(1, 1, 1);
        if (GUILayout.Button("Configure 4-8-6"))
            self.Configure(4, 8, 6);
        if (GUILayout.Button("Configure 8-128-1"))
            self.Configure(8, 128, 1);
        if (GUILayout.Button("Configure 1024-1024-1"))
            self.Configure(1024, 1024, 1);

        if (GUILayout.Button("Randomize"))
            self.Randomize();
        if (GUILayout.Button("RandomizeInputs"))
            self.RandomizeInputs();

        EditorGUILayout.Separator();

        showLayer0 = GUILayout.Toggle(showLayer0, "Show Layer 0");
        showLayer1 = GUILayout.Toggle(showLayer1, "Show Layer 1");
        showLayer2 = GUILayout.Toggle(showLayer2, "Show Layer 2");

        showWeights10 = GUILayout.Toggle(showWeights10, "Show Weights 1-0");
        showWeights21 = GUILayout.Toggle(showWeights21, "Show Weights 2-1");

        if (showLayer0)
        {
            EditorGUILayout.Separator();

            for (int i = 0; i < self.Layer0; ++i)
                GUILayout.Label(string.Format("L0: {0:00}: {1,7:F4} = {2,7:F4}", i, self.bias0[i], self.state0[i]));
        }

        if (showLayer1)
        {
            EditorGUILayout.Separator();

            for (int i = 0; i < self.Layer1; ++i)
                GUILayout.Label(string.Format("L1: {0:00}, {1,7:F4} = {2,7:F4}", i, self.bias1[i], self.state1[i]));
        }

        if (showLayer2)
        {
            EditorGUILayout.Separator();

            for (int i = 0; i < self.Layer2; ++i)
                GUILayout.Label(string.Format("L2: {0:00}, {1,7:F4} = {2,7:F4}", i, self.bias2[i], self.state2[i]));
        }

        if (showWeights10)
        {
            EditorGUILayout.Separator();

            for (int i = 0; i < self.Layer1; ++i)
            {
                var label = string.Format("L1: {0:00}, {1,7:F4}", i, self.bias1[i]);

                for (int j = 0; j < self.Layer0; ++j)
                    label += string.Format("{0,7:F3},", self.weights10[i * self.Layer0 + j]);

                GUILayout.Label(label);
            }
        }
        if (showWeights21)
        {
            EditorGUILayout.Separator();

            for (int i = 0; i < self.Layer2; ++i)
            {
                var label = string.Format("L2: {0:00}, {1,7:F4}", i, self.bias2[i]);

                for (int j = 0; j < self.Layer1; ++j)
                    label += string.Format("{0,7:F3},", self.weights21[i * self.Layer1 + j]);

                GUILayout.Label(label);
            }
        }

    }
}
#endif

public class NeuralBurst
    : MonoBehaviour
{
    [System.NonSerialized]
    public Rigidbody cachedRigidbody;

    private void OnEnable()
    {
        cachedRigidbody = GetComponent<Rigidbody>();
    }

    private void OnDestroy()
    {
        if (bias0.IsCreated) bias0.Dispose();
        if (bias1.IsCreated) bias1.Dispose();
        if (bias2.IsCreated) bias2.Dispose();

        if (state0.IsCreated) state0.Dispose();
        if (state1.IsCreated) state1.Dispose();
        if (state2.IsCreated) state2.Dispose();

        if (weights10.IsCreated) weights10.Dispose();
        if (weights21.IsCreated) weights21.Dispose();
    }

    private static float rand01() { return UnityEngine.Random.Range(0.0f, 1.0f); }
    private static float randh() { return UnityEngine.Random.Range(-0.5f, 0.5f); }
    private static float sigm(float x) { return x / (1.0f - Mathf.Exp(-x)); }
    private static float deriv(float x) { return sigm(x) * (1.0f - sigm(x)); }
    private static float tanh_slow(float x) { return (float)System.Math.Tanh((float)x); }

    // tanh series expansion approximation
    // https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
    private static float tanh(float x)
    {
        var xx = x * x;
        var ax = (((xx + 378.0f) * xx + 17325.0f) * xx + 135135.0f) * x;
        var bx = ((28.0f * xx + 3150.0f) * xx + 62370.0f) * xx + 135135;
        return ax / bx;
    }

    [System.NonSerialized] public bool Automatic;

    [System.NonSerialized] public int Layer0 = 4;
    [System.NonSerialized] public int Layer1 = 8;
    [System.NonSerialized] public int Layer2 = 6;

    [System.NonSerialized] public NativeArray<float> bias0;
    [System.NonSerialized] public NativeArray<float> bias1;
    [System.NonSerialized] public NativeArray<float> bias2;

    [System.NonSerialized] public NativeArray<float> state0;
    [System.NonSerialized] public NativeArray<float> state1;
    [System.NonSerialized] public NativeArray<float> state2;

    [System.NonSerialized] public NativeArray<float> weights10;
    [System.NonSerialized] public NativeArray<float> weights21;

    public void Configure(int layer0, int layer1, int layer2)
    {
        //Debug.LogFormat("Neural.Configure(): {0}:{1}:{2}", layer0, layer1, layer2);

        Layer0 = layer0;
        Layer1 = layer1;
        Layer2 = layer2;

        if (bias0.IsCreated) bias0.Dispose();
        if (bias1.IsCreated) bias1.Dispose();
        if (bias2.IsCreated) bias2.Dispose();

        if (state0.IsCreated) state0.Dispose();
        if (state1.IsCreated) state1.Dispose();
        if (state2.IsCreated) state2.Dispose();

        if (weights10.IsCreated) weights10.Dispose();
        if (weights21.IsCreated) weights21.Dispose();

        if (!bias0.IsCreated) bias0 = new NativeArray<float>(Layer0, Allocator.Persistent);
        if (!bias1.IsCreated) bias1 = new NativeArray<float>(Layer1, Allocator.Persistent);
        if (!bias2.IsCreated) bias2 = new NativeArray<float>(Layer2, Allocator.Persistent);

        if (!state0.IsCreated) state0 = new NativeArray<float>(Layer0, Allocator.Persistent);
        if (!state1.IsCreated) state1 = new NativeArray<float>(Layer1, Allocator.Persistent);
        if (!state2.IsCreated) state2 = new NativeArray<float>(Layer2, Allocator.Persistent);

        // [h0: w0..n], [h1: w0..wn]
        if (!weights10.IsCreated) weights10 = new NativeArray<float>(Layer1 * Layer0, Allocator.Persistent);
        if (!weights21.IsCreated) weights21 = new NativeArray<float>(Layer2 * Layer1, Allocator.Persistent);
    }

    public void Randomize()
    {
        //Debug.LogFormat("Neural.Randomize()");

        // randomize biases
        for (int i = 0; i < Layer0; ++i)
            bias0[i] = randh();
        for (int i = 0; i < Layer1; ++i)
            bias1[i] = randh();
        for (int i = 0; i < Layer2; ++i)
            bias2[i] = randh();

        // randomize weights
        for (int i = 0; i < Layer1; ++i)
            for (int j = 0; j < Layer0; ++j)
                weights10[i * Layer0 + j] = randh();

        // randomize weights
        for (int i = 0; i < Layer2; ++i)
            for (int j = 0; j < Layer1; ++j)
                weights21[i * Layer1 + j] = randh();
    }

    public void RandomizeInputs()
    {
        //Debug.LogFormat("Neural.RandomizeInputs()");

        // layer 0 (inputs)
        for (int i = 0; i < Layer0; ++i)
            state0[i] = rand01();
    }

    public void SetInputs(float[] inputs)
    {
        //Debug.LogFormat("Neural.SetInputs()");

        for (int i = 0; i < Layer0; ++i)
            state0[i] = inputs[i];
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct StepJob
        : IJob
    {
        [ReadOnly] public int Layer0;
        [ReadOnly] public int Layer1;
        [ReadOnly] public int Layer2;
        [ReadOnly] public NativeArray<float> bias0;
        [ReadOnly] public NativeArray<float> bias1;
        [ReadOnly] public NativeArray<float> bias2;
        [ReadOnly] public NativeArray<float> state0;
        public NativeArray<float> state1;
        public NativeArray<float> state2;
        [ReadOnly] public NativeArray<float> weights10;
        [ReadOnly] public NativeArray<float> weights21;

        public void Execute()
        {
            // layer 1 (hidden)
            for (int i = 0; i < Layer1; ++i)
            {
                var bias = bias1[i];
                var sum = 0.0f;
                for (int j = 0; j < Layer0; ++j)
                    sum += (state0[j] + bias0[j]) * weights10[i * Layer0 + j];
                state1[i] = tanh(bias + sum);
            }

            // layer 2 (output)
            for (int i = 0; i < Layer2; ++i)
            {
                var bias = bias2[i];
                var sum = 0.0f;
                for (int j = 0; j < Layer1; ++j)
                    sum += (state1[j] + bias1[j]) * weights21[i * Layer1 + j];
                state2[i] = tanh(bias + sum);
            }
        }
    }

    public StepJob QueueJobStep()
    {
        var job = new StepJob()
        {
            Layer0 = this.Layer0,
            Layer1 = this.Layer1,
            Layer2 = this.Layer2,
            bias0 = this.bias0,
            bias1 = this.bias1,
            bias2 = this.bias2,
            state0 = this.state0,
            state1 = this.state1,
            state2 = this.state2,
            weights10 = this.weights10,
            weights21 = this.weights21,
        };

        return job;
    }

    public void BackProp()
    {
        //Debug.LogFormat("Neural.BackProp(): running...");
    }


    public void LerpTowards(NeuralBurst from, float x)
    {
        for (int i = 0; i < Layer0; ++i)
            bias0[i] = Mathf.Lerp(bias0[i], from.bias0[i], x);
        for (int i = 0; i < Layer1; ++i)
            bias1[i] = Mathf.Lerp(bias1[i], from.bias1[i], x);
        for (int i = 0; i < Layer2; ++i)
            bias2[i] = Mathf.Lerp(bias2[i], from.bias2[i], x);

        for (int i = 0; i < Layer0 * Layer1; ++i)
            weights10[i] = Mathf.Lerp(weights10[i], from.weights10[i], x);
        for (int i = 0; i < Layer1 * Layer2; ++i)
            weights21[i] = Mathf.Lerp(weights21[i], from.weights21[i], x);
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct MutateJob
        : IJob
    {
        [ReadOnly] public uint Seed;
        [ReadOnly] public float MutateRate;
        [ReadOnly] public int Layer0;
        [ReadOnly] public int Layer1;
        [ReadOnly] public int Layer2;
        public NativeArray<float> bias0;
        public NativeArray<float> bias1;
        public NativeArray<float> bias2;
        public NativeArray<float> weights10;
        public NativeArray<float> weights21;

        public void Execute()
        {
            var r = new Unity.Mathematics.Random(Seed);

            for (int i = 0; i < Layer0; ++i)
                bias0[i] = math.select(bias0[i], r.NextFloat(-0.5f, 0.5f), r.NextFloat() < MutateRate);
            for (int i = 0; i < Layer1; ++i)
                bias1[i] = math.select(bias1[i], r.NextFloat(-0.5f, 0.5f), r.NextFloat() < MutateRate);
            for (int i = 0; i < Layer2; ++i)
                bias2[i] = math.select(bias2[i], r.NextFloat(-0.5f, 0.5f), r.NextFloat() < MutateRate);

            for (int i = 0; i < Layer0 * Layer1; ++i)
                weights10[i] = math.select(weights10[i], r.NextFloat(-0.5f, 0.5f), r.NextFloat() < MutateRate);
            for (int i = 0; i < Layer1 * Layer2; ++i)
                weights21[i] = math.select(weights21[i], r.NextFloat(-0.5f, 0.5f), r.NextFloat() < MutateRate);
        }
    }

    public MutateJob QueueJobMutate(uint seed, float mutateRate)
    {
        var job = new MutateJob()
        {
            Seed = seed,
            MutateRate = mutateRate,
            Layer0 = this.Layer0,
            Layer1 = this.Layer1,
            Layer2 = this.Layer2,
            bias0 = this.bias0,
            bias1 = this.bias1,
            bias2 = this.bias2,
            weights10 = this.weights10,
            weights21 = this.weights21,
        };

        return job;
    }

    public void Mutate(float x)
    {
        for (int i = 0; i < Layer0; ++i)
            bias0[i] = UnityEngine.Random.Range(0.0f, 1.0f) < x ? Mathf.Lerp(bias0[i], randh(), x) : bias0[i];
        for (int i = 0; i < Layer1; ++i)
            bias1[i] = UnityEngine.Random.Range(0.0f, 1.0f) < x ? Mathf.Lerp(bias1[i], randh(), x) : bias1[i];
        for (int i = 0; i < Layer2; ++i)
            bias2[i] = UnityEngine.Random.Range(0.0f, 1.0f) < x ? Mathf.Lerp(bias2[i], randh(), x) : bias2[i];

        for (int i = 0; i < Layer0 * Layer1; ++i)
            weights10[i] = UnityEngine.Random.Range(0.0f, 1.0f) < x ? Mathf.Lerp(weights10[i], randh(), x) : weights10[i];
        for (int i = 0; i < Layer1 * Layer2; ++i)
            weights21[i] = UnityEngine.Random.Range(0.0f, 1.0f) < x ? Mathf.Lerp(weights21[i], randh(), x) : weights21[i];
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct EvolveJob
        : IJob
    {
        [ReadOnly] public uint Seed;

        [ReadOnly] public float EvolveRate;

        [ReadOnly] public int Layer0;
        [ReadOnly] public int Layer1;
        [ReadOnly] public int Layer2;

        [ReadOnly] public NativeArray<float> from_bias0;
        [ReadOnly] public NativeArray<float> from_bias1;
        [ReadOnly] public NativeArray<float> from_bias2;
        [ReadOnly] public NativeArray<float> from_weights10;
        [ReadOnly] public NativeArray<float> from_weights21;

        public NativeArray<float> bias0;
        public NativeArray<float> bias1;
        public NativeArray<float> bias2;
        public NativeArray<float> weights10;
        public NativeArray<float> weights21;

        public void Execute()
        {
            var r = new Unity.Mathematics.Random(Seed);

            for (int i = 0; i < Layer0; ++i)
                bias0[i] = math.select(bias0[i], from_bias0[i], r.NextFloat() < EvolveRate);
            for (int i = 0; i < Layer1; ++i)
                bias1[i] = math.select(bias1[i], from_bias1[i], r.NextFloat() < EvolveRate);
            for (int i = 0; i < Layer2; ++i)
                bias2[i] = math.select(bias2[i], from_bias2[i], r.NextFloat() < EvolveRate);

            for (int i = 0; i < Layer0 * Layer1; ++i)
                weights10[i] = math.select(weights10[i], from_weights10[i], r.NextFloat() < EvolveRate);
            for (int i = 0; i < Layer1 * Layer2; ++i)
                weights21[i] = math.select(weights21[i], from_weights21[i], r.NextFloat() < EvolveRate);
        }
    }

    public EvolveJob QueueJobEvolve(uint seed, NeuralBurst from, float evolveRate)
    {
        var job = new EvolveJob()
        {
            Seed = seed,
            EvolveRate = evolveRate,
            Layer0 = this.Layer0,
            Layer1 = this.Layer1,
            Layer2 = this.Layer2,
            from_bias0 = from.bias0,
            from_bias1 = from.bias1,
            from_bias2 = from.bias2,
            from_weights10 = from.weights10,
            from_weights21 = from.weights21,
            bias0 = this.bias0,
            bias1 = this.bias1,
            bias2 = this.bias2,
            weights10 = this.weights10,
            weights21 = this.weights21,
        };

        return job;
    }

    public void Evolve(NeuralBurst from, float x)
    {
        for (int i = 0; i < Layer0; ++i)
            bias0[i] = UnityEngine.Random.Range(0.0f, 1.0f) < x ? from.bias0[i] : bias0[i];
        for (int i = 0; i < Layer1; ++i)
            bias1[i] = UnityEngine.Random.Range(0.0f, 1.0f) < x ? from.bias1[i] : bias1[i];
        for (int i = 0; i < Layer2; ++i)
            bias2[i] = UnityEngine.Random.Range(0.0f, 1.0f) < x ? from.bias2[i] : bias2[i];

        for (int i = 0; i < Layer0 * Layer1; ++i)
            weights10[i] = UnityEngine.Random.Range(0.0f, 1.0f) < x ? from.weights10[i] : weights10[i];
        for (int i = 0; i < Layer1 * Layer2; ++i)
            weights21[i] = UnityEngine.Random.Range(0.0f, 1.0f) < x ? from.weights21[i] : weights21[i];
    }

    public Color GenerateColor()
    {
        var result = new Color(0, 0, 0, 1);

        for (int i = 0; i < Layer0; ++i)
            result.r += bias0[i] + 0.5f;
        result.r /= (float)Layer0;

        for (int i = 0; i < Layer1; ++i)
            result.g += bias1[i] + 0.5f;
        result.g /= (float)Layer1;

        for (int i = 0; i < Layer2; ++i)
            result.b += bias2[i] + 0.5f;
        result.b /= (float)Layer2;

        return result;
    }
}
