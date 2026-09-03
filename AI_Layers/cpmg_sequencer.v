module cpmg_sequencer (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [15:0] tau,      // Programmable delay in clock cycles (e.g., 500 for 10us)
    input  wire [7:0]  n_pulses, // N ~ 50 pulses to fit within the ~1 ms T2 coherence budget
    output reg         mw_gate,  // Drives the microwave switch
    output reg         done      // Flags when the sequence is complete
);

    // State Machine Encodings
    localparam IDLE      = 3'd0;
    localparam PI_HALF_1 = 3'd1;
    localparam TAU_1     = 3'd2;
    localparam PI_PULSE  = 3'd3;
    localparam TAU_2     = 3'd4;
    localparam PI_HALF_2 = 3'd5;
    
    // Pulse widths in 20ns clock cycles (Placeholder values for 50MHz, will need PLL later)
    localparam WIDTH_PI_HALF = 16'd1; // 20 ns
    localparam WIDTH_PI      = 16'd2; // 40 ns

    reg [2:0]  state;
    reg [15:0] timer;
    reg [7:0]  pulse_count;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state       <= IDLE;
            mw_gate     <= 1'b0;
            done        <= 1'b0;
            timer       <= 16'd0;
            pulse_count <= 8'd0;
        end else begin
            case (state)
                IDLE: begin
                    done    <= 1'b0;
                    mw_gate <= 1'b0;
                    if (start) begin
                        state <= PI_HALF_1;
                        timer <= WIDTH_PI_HALF;
                    end
                end

                PI_HALF_1: begin
                    mw_gate <= 1'b1;
                    if (timer == 16'd1) begin
                        state <= TAU_1;
                        timer <= tau;
                        mw_gate <= 1'b0;
                    end else timer <= timer - 16'd1;
                end

                TAU_1: begin
                    if (timer == 16'd1) begin
                        state <= PI_PULSE;
                        timer <= WIDTH_PI;
                    end else timer <= timer - 16'd1;
                end

                PI_PULSE: begin
                    mw_gate <= 1'b1;
                    if (timer == 16'd1) begin
                        state <= TAU_2;
                        timer <= tau;
                        mw_gate <= 1'b0;
                    end else timer <= timer - 16'd1;
                end

                TAU_2: begin
                    if (timer == 16'd1) begin
                        pulse_count <= pulse_count + 8'd1;
                        if (pulse_count + 8'd1 >= n_pulses) begin
                            state <= PI_HALF_2;
                            timer <= WIDTH_PI_HALF;
                        end else begin
                            state <= TAU_1;
                            timer <= tau;
                        end
                    end else timer <= timer - 16'd1;
                end

                PI_HALF_2: begin
                    mw_gate <= 1'b1;
                    if (timer == 16'd1) begin
                        state <= IDLE;
                        done  <= 1'b1;
                        mw_gate <= 1'b0;
                        pulse_count <= 8'd0;
                    end else timer <= timer - 16'd1;
                end
                
                default: state <= IDLE;
            endcase
        end
    end
endmodule
